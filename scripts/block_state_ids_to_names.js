const fs = require('fs')

function parseArgs () {
  const args = process.argv.slice(2)
  const parsed = { version: '1.16.4' }
  for (let i = 0; i < args.length; i++) {
    const arg = args[i]
    if (!arg.startsWith('--')) continue
    parsed[arg.slice(2)] = args[i + 1]
    i++
  }
  if (!parsed.input || !parsed.output) {
    console.error('Usage: node scripts/block_state_ids_to_names.js --input ids.json --output map.json [--version 1.16.4]')
    process.exit(1)
  }
  return parsed
}

const args = parseArgs()
const Block = require('prismarine-block')(args.version)
const ids = JSON.parse(fs.readFileSync(args.input, 'utf8'))
const mapping = {}

for (const rawId of ids) {
  const id = Number(rawId)
  try {
    const block = Block.fromStateId(id, 0)
    mapping[id] = block && block.name ? block.name : `unknown_${id}`
  } catch (error) {
    mapping[id] = `unknown_${id}`
  }
}

fs.writeFileSync(args.output, JSON.stringify(mapping, null, 2))
