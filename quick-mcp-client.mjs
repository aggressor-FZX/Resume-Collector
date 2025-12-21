// quick-mcp-client.mjs
// Run: node quick-mcp-client.mjs
const { Client } = await import('@modelcontextprotocol/sdk/client/index.js');
const stdioModule = await import('@modelcontextprotocol/sdk/client/stdio.js');

async function main() {
  // Construct the stdio transport using the exported class
  const StdioClientTransport = stdioModule.StdioClientTransport;
  if (!StdioClientTransport) {
    console.error('StdioClientTransport not found in stdio module exports:', Object.keys(stdioModule));
    process.exit(1);
  }

  const transport = new StdioClientTransport({
    command: './node_modules/.bin/task-master-ai',
    args: ['--config', '.taskmasterrc.json']
  });

  // Create client with a name/version and the transport
  const clientInfo = { name: 'quick-mcp-client', version: '0.0.1' };
  const client = new Client(clientInfo);

  // Register capabilities the server expects so providers get registered
  try {
    client.registerCapabilities?.({
      sampling: { supportsSampling: true },
      streaming: { supportsStreaming: true },
      elicitation: { form: { applyDefaults: true } }
    });
  } catch (err) {
    console.warn('registerCapabilities warning:', err?.message ?? err);
  }

  // Connect and let the server discover capabilities
  await client.connect(transport);
  console.log('Client connected');

  // Keep alive briefly so server can register providers
  await new Promise((r) => setTimeout(r, 2000));

  // Disconnect if supported
  await client.close();
  console.log('Client disconnected');
}

main().catch((err) => {
  console.error('Client error:', err);
  process.exit(1);
});
