import argparse
import json
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MINECRAFT_DATA_VERSION_ALIASES = {
    "1.16.4": "1.16.2",
}


def collect_unique_state_ids(input_path, batch_size):
    parquet = pq.ParquetFile(input_path)
    state_ids = set()
    for batch in tqdm(
        parquet.iter_batches(batch_size=batch_size, columns=["voxel_data"]),
        total=parquet.metadata.num_row_groups,
        desc="Collecting block state ids",
    ):
        values = batch.column(0).values.to_numpy(zero_copy_only=False)
        state_ids.update(int(value) for value in np.unique(values))
    return sorted(state_ids)


def build_state_id_name_map_from_node(input_path, output_path, version, batch_size):
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    state_ids = collect_unique_state_ids(input_path, batch_size)
    with tempfile.TemporaryDirectory() as tmp_dir:
        ids_path = Path(tmp_dir) / "state_ids.json"
        ids_path.write_text(json.dumps(state_ids), encoding="utf-8")
        subprocess.run(
            [
                "node",
                str(SCRIPT_DIR / "block_state_ids_to_names.js"),
                "--input",
                str(ids_path),
                "--output",
                str(output_path),
                "--version",
                version,
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )

    with output_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_state_id_name_map_from_minecraft_data(output_path, version):
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    data_version = MINECRAFT_DATA_VERSION_ALIASES.get(version, version)
    url = f"https://raw.githubusercontent.com/PrismarineJS/minecraft-data/master/data/pc/{data_version}/blocks.json"
    with urllib.request.urlopen(url, timeout=60) as response:
        blocks = json.loads(response.read().decode("utf-8"))

    mapping = {}
    for block in blocks:
        name = block["name"]
        for state_id in range(int(block["minStateId"]), int(block["maxStateId"]) + 1):
            mapping[str(state_id)] = name

    output_path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    return mapping


def build_state_id_name_map(input_path, output_path, version, batch_size, mapping_source):
    if mapping_source == "node":
        return build_state_id_name_map_from_node(input_path, output_path, version, batch_size)
    if mapping_source == "minecraft-data-json":
        return build_state_id_name_map_from_minecraft_data(output_path, version)
    if mapping_source != "auto":
        raise ValueError(f"Unknown mapping source: {mapping_source}")

    try:
        return build_state_id_name_map_from_node(input_path, output_path, version, batch_size)
    except Exception as error:
        print(f"Node mapping failed ({error}); falling back to minecraft-data blocks.json")
        return build_state_id_name_map_from_minecraft_data(output_path, version)


def append_unique_names(table, state_id_to_name):
    voxel_column = table.column("voxel_data")
    unique_ids_by_row = []
    unique_names_by_row = []

    for voxel_data in voxel_column.to_pylist():
        unique_ids = sorted({int(value) for value in voxel_data if int(value) != 0})
        unique_ids_by_row.append(unique_ids)
        unique_names_by_row.append([state_id_to_name.get(str(value), f"unknown_{value}") for value in unique_ids])

    table = table.append_column(
        "unique_block_state_ids",
        pa.array(unique_ids_by_row, type=pa.list_(pa.int64())),
    )
    table = table.append_column(
        "unique_block_names",
        pa.array(unique_names_by_row, type=pa.list_(pa.string())),
    )
    return table


def append_voxel_names(table, state_id_to_name, drop_voxel_data):
    voxel_column = table.column("voxel_data")
    names_by_row = []

    for voxel_data in voxel_column.to_pylist():
        names_by_row.append([state_id_to_name.get(str(int(value)), f"unknown_{int(value)}") for value in voxel_data])

    table = table.append_column(
        "voxel_name_data",
        pa.array(names_by_row, type=pa.list_(pa.string())),
    )
    if drop_voxel_data:
        table = table.drop(["voxel_data"])
    return table


def write_output_parquet(input_path, output_path, state_id_to_name, mode, batch_size, drop_voxel_data):
    parquet = pq.ParquetFile(input_path)
    writer = None

    try:
        for batch in tqdm(
            parquet.iter_batches(batch_size=batch_size),
            total=(parquet.metadata.num_rows + batch_size - 1) // batch_size,
            desc=f"Writing {mode} parquet",
        ):
            table = pa.Table.from_batches([batch])
            if mode == "unique-names":
                table = append_unique_names(table, state_id_to_name)
            elif mode == "voxel-names":
                table = append_voxel_names(table, state_id_to_name, drop_voxel_data)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Add Minecraft block names derived from voxel_data block state ids.")
    parser.add_argument("--input", default=str(PROJECT_ROOT / "data (1).parquet"), help="Input Parquet path.")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "data_with_block_names.parquet"), help="Output Parquet path.")
    parser.add_argument("--map-output", default=str(PROJECT_ROOT / "block_state_id_to_name.json"), help="Output JSON mapping path.")
    parser.add_argument("--version", default="1.16.4", help="Minecraft version for prismarine-block.")
    parser.add_argument(
        "--mapping-source",
        choices=("auto", "node", "minecraft-data-json"),
        default="auto",
        help="How to build block state id names. minecraft-data-json avoids npm/node_modules.",
    )
    parser.add_argument(
        "--mode",
        choices=("map", "unique-names", "voxel-names"),
        default="unique-names",
        help="map only writes JSON mapping; unique-names adds compact unique names per row; voxel-names adds a full 32768-name list per row.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Parquet batch size.")
    parser.add_argument(
        "--drop-voxel-data",
        action="store_true",
        help="Only for --mode voxel-names: drop numeric voxel_data from the output to reduce size.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    map_output = Path(args.map_output)

    state_id_to_name = build_state_id_name_map(
        input_path,
        map_output,
        args.version,
        args.batch_size,
        args.mapping_source,
    )
    print(f"Wrote/loaded {len(state_id_to_name)} block state mappings: {map_output}")

    if args.mode == "map":
        return

    write_output_parquet(
        input_path=input_path,
        output_path=output_path,
        state_id_to_name=state_id_to_name,
        mode=args.mode,
        batch_size=args.batch_size,
        drop_voxel_data=args.drop_voxel_data,
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
