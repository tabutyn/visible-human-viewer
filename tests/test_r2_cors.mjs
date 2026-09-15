import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

test("R2 CORS policy uses the Cloudflare API schema", async () => {
  const policy = JSON.parse(
    await readFile(new URL("../r2-cors.json", import.meta.url), "utf8"),
  );

  assert.ok(Array.isArray(policy.rules));
  assert.equal(policy.rules.length, 1);

  const [rule] = policy.rules;
  assert.deepEqual(rule.allowed.methods, ["GET", "HEAD"]);
  assert.ok(
    rule.allowed.origins.includes("https://visiblehuman.ballrollergames.com"),
  );
  assert.deepEqual(rule.exposeHeaders, [
    "Content-Length",
    "Content-Encoding",
    "ETag",
  ]);
  assert.equal(rule.maxAgeSeconds, 86400);

  for (const legacyKey of [
    "AllowedOrigins",
    "AllowedMethods",
    "AllowedHeaders",
    "ExposeHeaders",
    "MaxAgeSeconds",
  ]) {
    assert.equal(legacyKey in rule, false);
  }
});
