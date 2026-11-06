const fs = require('node:fs');
const https = require('node:https');
const path = require('node:path');
const {spawnSync} = require('node:child_process');

if (!process.argv.includes('--child')) {
  const env = {...process.env};
  for (const key of ['NODE_OPTIONS', 'NODE_EXTRA_CA_CERTS', 'SSL_CERT_FILE', 'SSL_CERT_DIR', 'NODE_TLS_REJECT_UNAUTHORIZED']) delete env[key];
  for (const key of Object.keys(env)) if (key.toLowerCase().endsWith('_proxy')) delete env[key];
  const result = spawnSync(process.execPath, [__filename, '--child'], {env, stdio: 'inherit', timeout: 180000});
  if (result.error) throw result.error;
  process.exit(result.status ?? 1);
}

const hosts = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'hosts.json')));
const controls = ['github.com', 'pypi.org', 'files.pythonhosted.org', 'probe-emudhra.we-have-a-problem.com'];
const rejected = new Set(['UNABLE_TO_GET_ISSUER_CERT', 'UNABLE_TO_GET_ISSUER_CERT_LOCALLY', 'UNABLE_TO_VERIFY_LEAF_SIGNATURE', 'SELF_SIGNED_CERT_IN_CHAIN', 'CERT_REVOKED', 'CERT_UNTRUSTED', 'CERT_REJECTED']);
const emit = value => console.log('TLS_DIAGNOSTIC ' + JSON.stringify(value));
emit({kind: 'metadata', label: 'runner-embedded-node', executable: process.execPath, versions: process.versions, argv: process.execArgv});
fs.writeFileSync(path.join(process.env.RUNNER_TEMP, 'node-roots.pem'), require('node:tls').rootCertificates.join('\n'));

function probe(host) {
  return new Promise(resolve => {
    let done = false;
    const finish = (status, detail) => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      resolve({kind: 'result', label: 'runner-embedded-node', client: 'node', host, status, expected: controls.includes(host) ? 'connected' : 'rejected', detail});
    };
    const request = https.get(`https://${host}/`, {headers: {'User-Agent': `MNT-temporary-TLS-diagnostic Node/${process.versions.node}`}}, response => {
      let body = '';
      response.on('data', chunk => { if (body.length < 8192) body += chunk; });
      response.on('end', () => finish(controls.includes(host) || body.includes('Your client successfully connected to the site.') ? 'connected' : 'inconclusive', `HTTP ${response.statusCode}`));
      response.on('error', error => finish('inconclusive', error.message));
    });
    const timer = setTimeout(() => { finish('inconclusive', 'timeout'); request.destroy(); }, 20000);
    request.on('error', error => finish(rejected.has(error.code) || /compromised|blocked public key|public key is blocked/i.test(error.message) ? 'rejected' : 'inconclusive', `${error.code}: ${error.message}`));
  });
}

(async () => {
  const results = [];
  const all = [...controls, ...hosts];
  for (let i = 0; i < all.length; i += 3) {
    for (const result of await Promise.all(all.slice(i, i + 3).map(probe))) {
      results.push(result);
      emit(result);
    }
  }
  const nonpassing = results.filter(result => result.status !== result.expected).length;
  emit({kind: 'summary', label: 'runner-embedded-node', measurements: results.length, nonpassing});
  fs.appendFileSync(process.env.GITHUB_STEP_SUMMARY, `\n### Runner embedded Node ${process.versions.node}\n\n${results.length} measurements; **${nonpassing} unsafe or inconclusive**.\n`);
})().catch(error => { console.error(error); process.exitCode = 1; });
