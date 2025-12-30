// list-tools.mjs
// Run: node list-tools.mjs
const { Client } = await import('@modelcontextprotocol/sdk/client/index.js');
const stdioModule = await import('@modelcontextprotocol/sdk/client/stdio.js').catch(() => null);
if (!stdioModule) {
  console.error('Stdio client module not found in SDK. Inspect node_modules/@modelcontextprotocol/sdk/dist/esm/client');
  process.exit(1);
}

const StdioClientTransport = stdioModule.StdioClientTransport;
if (!StdioClientTransport) {
  console.error('Stdio transport not found in stdio module exports:', Object.keys(stdioModule));
  process.exit(1);
}

// Spawn the Task Master AI MCP server and connect via stdio so we can list tools
const transport = new StdioClientTransport({
  command: './node_modules/.bin/task-master-ai',
  args: ['--config', '.taskmasterrc.json']
});

const clientInfo = { name: 'tool-list-client', version: '0.0.1' };
const client = new Client(clientInfo);

// Connect using the transport instance to ensure Client._transport is set
await client.connect(transport);
try {
  const tools = await client.listTools();
  console.log('Registered tools:', JSON.stringify(tools, null, 2));
} catch (err) {
  console.error('Error listing tools:', err);
} finally {
  await client.close?.();
}
process.exit(0);
