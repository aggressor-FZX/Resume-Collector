// invoke-tool.mjs
// Usage: node invoke-tool.mjs <toolName> '<jsonArguments>'
const { Client } = await import('@modelcontextprotocol/sdk/client/index.js');
const stdioModule = await import('@modelcontextprotocol/sdk/client/stdio.js');

function usage() {
  console.error('Usage: node invoke-tool.mjs <toolName> "<jsonArguments>"');
  console.error('Example: node invoke-tool.mjs parse_prd "{\"input\":\"/workspaces/Resume-Collector/PRD_Data_Ingestion.md\",\"projectRoot\":\"/workspaces/Resume-Collector\",\"force\":true}"');
  process.exit(1);
}

if (process.argv.length < 3) usage();

const toolName = process.argv[2];
let args = {};
if (process.argv[3]) {
  try {
    args = JSON.parse(process.argv[3]);
  } catch (err) {
    console.error('Invalid JSON in arguments:', err.message);
    usage();
  }
}

async function main() {
  const StdioClientTransport = stdioModule.StdioClientTransport;
  if (!StdioClientTransport) {
    console.error('StdioClientTransport not found in stdio module exports:', Object.keys(stdioModule));
    process.exit(1);
  }

  const transport = new StdioClientTransport({
    command: './node_modules/.bin/task-master-ai',
    args: ['--config', '.taskmasterrc.json']
  });

  const clientInfo = { name: 'invoke-tool-client', version: '0.0.1' };
  const client = new Client(clientInfo);

  try {
    await client.connect(transport);
    console.log('Client connected');

    const res = await client.callTool({ name: toolName, arguments: args });
    console.log('Tool result:', JSON.stringify(res, null, 2));
  } catch (err) {
    console.error('Error invoking tool:', err);
    process.exitCode = 1;
  } finally {
    await client.close?.();
    console.log('Client disconnected');
  }
}

main().catch((err) => {
  console.error('Client error:', err);
  process.exit(1);
});
