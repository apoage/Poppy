#!/usr/bin/env node
/**
 * Claude Code Router
 * - Haiku requests → Cerebras GLM-4.7 (OpenAI format, translated)
 * - Everything else → Anthropic (passthrough)
 *
 * Usage:
 *   CEREBRAS_API_KEY=csk-xxx node tools/claude-router.js
 *
 * Then in ~/.claude/settings.json set:
 *   "env": { "ANTHROPIC_BASE_URL": "http://127.0.0.1:8082" }
 *
 * In Claude Code, /model claude-haiku-4-5-20251001 → GLM
 *                  /model claude-opus-4-6            → Anthropic
 */

const http = require('http');
const https = require('https');

const PORT = process.env.ROUTER_PORT || 8082;
const CEREBRAS_KEY = process.env.CEREBRAS_API_KEY;
const ANTHROPIC_KEY = process.env.ANTHROPIC_API_KEY;

if (!CEREBRAS_KEY) {
  console.error('ERROR: CEREBRAS_API_KEY environment variable required');
  process.exit(1);
}

// Models routed to Cerebras GLM
const GLM_MODELS = new Set([
  'claude-haiku-4-5-20251001',
  'haiku',
]);

// ── Anthropic → OpenAI message format translation ──

function translateMessages(anthropicMessages) {
  return anthropicMessages.map(msg => {
    // Simple text message
    if (typeof msg.content === 'string') {
      return { role: msg.role, content: msg.content };
    }
    // Content blocks array
    if (Array.isArray(msg.content)) {
      const textParts = msg.content
        .filter(b => b.type === 'text')
        .map(b => b.text);
      return { role: msg.role, content: textParts.join('\n') || '' };
    }
    return { role: msg.role, content: '' };
  });
}

function anthropicToOpenAI(body) {
  const messages = [];

  // System prompt
  if (body.system) {
    const sysText = typeof body.system === 'string'
      ? body.system
      : Array.isArray(body.system)
        ? body.system.map(b => b.text || '').join('\n')
        : '';
    if (sysText) {
      messages.push({ role: 'system', content: sysText });
    }
  }

  // Conversation messages
  if (body.messages) {
    messages.push(...translateMessages(body.messages));
  }

  return {
    model: 'zai-glm-4.7',
    messages,
    max_tokens: body.max_tokens || 4096,
    temperature: 1,
    top_p: 0.95,
    stream: !!body.stream,
  };
}

// ── OpenAI SSE → Anthropic SSE translation ──

function openaiChunkToAnthropic(chunk) {
  // Parse OpenAI chunk
  let data;
  try { data = JSON.parse(chunk); } catch { return null; }

  const choice = data.choices?.[0];
  if (!choice) return null;

  const delta = choice.delta || {};

  // Stop event
  if (choice.finish_reason) {
    return [
      jsonSSE('content_block_stop', { type: 'content_block_stop', index: 0 }),
      jsonSSE('message_delta', {
        type: 'message_delta',
        delta: { stop_reason: 'end_turn', stop_sequence: null },
        usage: { output_tokens: data.usage?.completion_tokens || 0 },
      }),
      jsonSSE('message_stop', { type: 'message_stop' }),
    ];
  }

  // Text delta
  if (delta.content) {
    return [
      jsonSSE('content_block_delta', {
        type: 'content_block_delta',
        index: 0,
        delta: { type: 'text_delta', text: delta.content },
      }),
    ];
  }

  return null;
}

function jsonSSE(eventType, data) {
  return `event: ${eventType}\ndata: ${JSON.stringify(data)}\n\n`;
}

// ── Proxy handler ──

const server = http.createServer((clientReq, clientRes) => {
  const chunks = [];
  clientReq.on('data', chunk => chunks.push(chunk));
  clientReq.on('end', () => {
    const rawBody = Buffer.concat(chunks);

    let body = {};
    let model = '';
    try {
      body = JSON.parse(rawBody);
      model = body.model || '';
    } catch {}

    const isGlm = GLM_MODELS.has(model);

    if (isGlm) {
      handleGLM(body, clientReq, clientRes);
    } else {
      handleAnthropic(rawBody, clientReq, clientRes);
    }
  });
});

// ── GLM path: translate and proxy to Cerebras ──

function handleGLM(body, clientReq, clientRes) {
  const openaiBody = anthropicToOpenAI(body);
  const payload = JSON.stringify(openaiBody);
  const isStream = openaiBody.stream;

  console.log(`[${ts()}] GLM ← ${body.model} (stream=${isStream})`);

  const options = {
    hostname: 'api.cerebras.ai',
    port: 443,
    path: '/v1/chat/completions',
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${CEREBRAS_KEY}`,
      'Content-Length': Buffer.byteLength(payload),
    },
  };

  const proxyReq = https.request(options, (proxyRes) => {
    if (!isStream) {
      // Non-streaming: collect full response and translate
      const resChunks = [];
      proxyRes.on('data', c => resChunks.push(c));
      proxyRes.on('end', () => {
        try {
          const oaiRes = JSON.parse(Buffer.concat(resChunks).toString());
          const anthropicRes = {
            id: oaiRes.id || 'msg_glm',
            type: 'message',
            role: 'assistant',
            model: 'glm-4.7',
            content: [{
              type: 'text',
              text: oaiRes.choices?.[0]?.message?.content || '',
            }],
            stop_reason: 'end_turn',
            usage: {
              input_tokens: oaiRes.usage?.prompt_tokens || 0,
              output_tokens: oaiRes.usage?.completion_tokens || 0,
            },
          };
          clientRes.writeHead(200, { 'Content-Type': 'application/json' });
          clientRes.end(JSON.stringify(anthropicRes));
        } catch (e) {
          console.error(`[GLM] Parse error: ${e.message}`);
          clientRes.writeHead(502, { 'Content-Type': 'application/json' });
          clientRes.end(JSON.stringify({ error: e.message }));
        }
      });
      return;
    }

    // Streaming: translate SSE chunks
    clientRes.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
    });

    // Send Anthropic stream start events
    clientRes.write(jsonSSE('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_glm_' + Date.now(),
        type: 'message',
        role: 'assistant',
        model: 'glm-4.7',
        content: [],
        stop_reason: null,
        usage: { input_tokens: 0, output_tokens: 0 },
      },
    }));
    clientRes.write(jsonSSE('content_block_start', {
      type: 'content_block_start',
      index: 0,
      content_block: { type: 'text', text: '' },
    }));

    let buffer = '';
    proxyRes.on('data', (chunk) => {
      buffer += chunk.toString();
      const lines = buffer.split('\n');
      buffer = lines.pop(); // keep incomplete line

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const data = line.slice(6).trim();
        if (data === '[DONE]') {
          // Send stop events
          clientRes.write(jsonSSE('content_block_stop', { type: 'content_block_stop', index: 0 }));
          clientRes.write(jsonSSE('message_delta', {
            type: 'message_delta',
            delta: { stop_reason: 'end_turn', stop_sequence: null },
            usage: { output_tokens: 0 },
          }));
          clientRes.write(jsonSSE('message_stop', { type: 'message_stop' }));
          clientRes.end();
          return;
        }

        const events = openaiChunkToAnthropic(data);
        if (events) {
          for (const ev of events) {
            clientRes.write(ev);
          }
        }
      }
    });

    proxyRes.on('end', () => {
      if (!clientRes.writableEnded) clientRes.end();
    });
  });

  proxyReq.on('error', (e) => {
    console.error(`[GLM] Proxy error: ${e.message}`);
    clientRes.writeHead(502, { 'Content-Type': 'application/json' });
    clientRes.end(JSON.stringify({ error: `GLM proxy error: ${e.message}` }));
  });

  proxyReq.write(payload);
  proxyReq.end();
}

// ── Anthropic path: straight passthrough ──

function handleAnthropic(rawBody, clientReq, clientRes) {
  let model = '';
  try { model = JSON.parse(rawBody).model || ''; } catch {}

  console.log(`[${ts()}] Anthropic ← ${model} ${clientReq.method} ${clientReq.url}`);

  const outHeaders = { ...clientReq.headers };
  delete outHeaders['host'];
  outHeaders['host'] = 'api.anthropic.com';

  const options = {
    hostname: 'api.anthropic.com',
    port: 443,
    path: clientReq.url,
    method: clientReq.method,
    headers: outHeaders,
  };

  const proxyReq = https.request(options, (proxyRes) => {
    clientRes.writeHead(proxyRes.statusCode, proxyRes.headers);
    proxyRes.pipe(clientRes);
  });

  proxyReq.on('error', (e) => {
    console.error(`[Anthropic] Proxy error: ${e.message}`);
    clientRes.writeHead(502, { 'Content-Type': 'application/json' });
    clientRes.end(JSON.stringify({ error: `Proxy error: ${e.message}` }));
  });

  proxyReq.write(rawBody);
  proxyReq.end();
}

function ts() { return new Date().toISOString(); }

server.listen(PORT, '127.0.0.1', () => {
  console.log(`Claude Code Router on http://127.0.0.1:${PORT}`);
  console.log(`  Haiku (/model claude-haiku-4-5-20251001) → GLM-4.7 via Cerebras`);
  console.log(`  Opus/Sonnet → Anthropic (passthrough)`);
});
