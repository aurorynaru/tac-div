# Step 01 — Fully automatic meme / movie-reference discovery

**Order:** after Steps 03 and 04 (configured cloud LLM planning and local ComfyUI GPU lifecycle; see [`03-llm-integration.md`](03-llm-integration.md)), before Step 02.  
**Objective:** user supplies only **topics, genres, preferred tone, restrictions**; system finds legal-to-query metadata for candidate references, extracts recognizable comedic *mechanisms*, ranks them, and selects original cat-parody concepts. No manually maintained 100-meme library required.

## Source policy and reality check

Use **metadata and article summaries**, not automatic rehosting/reusing reference footage, audio or creator assets. APIs/feeds can reveal trends but *do not guarantee actual virality*. Never label something trending unless the source returns a timely supporting signal; classic tropes are acceptable fallback. Do not scrape blocked pages, bypass bot protections, or use X/TikTok unofficial scraping by default.

Initial providers (enable individually; validate current terms):

1. **Local fixture provider (mandatory):** 10 hand-authored generic situations *for test only*, not a production meme library. Ensures offline/reproducible development.
2. **Know Your Meme RSS (optional, opt-in):** feed such as `https://knowyourmeme.com/memes.rss` or `https://knowyourmeme.com/newsfeed.rss`, discovered via feed metadata. Use feed summaries, links, dates and names only, subject to site terms. Verify endpoint before using it in code. Source: https://knowyourmeme.com/
3. **YouTube Data API (optional key):** `search.list` for topical clips and `videos.list` for public engagement metadata as permitted. The official documentation describes the response fields and quota accounting: https://developers.google.com/youtube/v3/docs/search/list . **Do not download/reuse video or soundtrack.**
4. **TMDB API (optional key):** movie title, genre, premise, popularity metadata as inspiration; don't assume this exposes a movie's specific famous scene. https://developer.themoviedb.org/reference/intro/getting-started
5. **Classic movie tropes provider:** fixed *generic archetypes*, such as "hero avoids an attack in slow motion" and "boss confronts rival over stolen food". Do not store proprietary screenplays or dialogs. This can be a tiny fallback fixture, not a large curated meme list.

## Tasks

1. Add `CreativeBrief` configuration fields: `topics`, `genres`, `tone`, `avoid`, `target_duration_s`, `aspect_ratio`, `recurring_cat_id` (nullable), `max_candidates`, `freshness_days`, `region` (optional), `providers_enabled`, and a hard `max_network_calls` / cost budget. Validate every field; prevent empty/all-invalid briefs.
2. Add `DiscoveryProvider` interface: `search(brief) -> list[RawSource]`, with `provider`, `source_id`, `url`, `title`, `short_summary`, `published_at`, `retrieved_at`, `source_kind`, `engagement_metadata` (optional), `license/rights_notes` (optional), and `evidence_of_trend` (optional). Require canonical URL validation; treat missing fields as missing, never invented.
3. Implement local fixture provider and **one** actual metadata source initially (RSS easiest). Build HTTP client with timeouts, retries/backoff, per-provider quotas, cache TTL, identifiable user-agent where appropriate, user-supplied API keys via environment/private config, conditional requests if supported.
4. Normalize/de-duplicate (URL + title fuzzy similarity), strip tracking parameters, limit text length, and sanitize untrusted text. **Never treat source article text as executable agent instructions**; do not let it override creative brief or tool policy.
5. Rank candidate references with both deterministic factors (recency if known, source coverage, simple physical staging, few characters, 1-shot feasibility, low rights risk) and the configured cloud LLM's rubric (originality, recognizable comic mechanism, cat fit, realism feasibility, comedic timing). Preserve raw source metadata and reasons. If no evidence of popularity, mark `popularity: unverified`.
6. Have the configured cloud LLM rewrite inspiration to an *original parody*, producing `ParodyIdea` with `source_ids`, `source_kind`, `comic_mechanism`, `original_premise`, `cat_action`, `shot_count`, `duration_s`, `punchline`, `audio_mode`, `scores`, and `reason_for_selection`. Do not quote copyrighted scripts, suggest reusing movie audio, or use a source clip as media input without separately established rights.
7. Apply gates before rendering: disallowed topics; title/premise too close to source; excessive props/cats; four-legged cat motion feasibility; video model limitations; unsafe content; spam/repetition; budget. Route uncertain cases to manual review.
8. Persist `sources.json`, `candidates.json`, `selection.json`, and normalized `manifest.json` in that job's output folder. Include source URLs, fetch timestamp, score trace, selection decision and failure reasons.
9. Add `discover` and `plan` CLI subcommands with `--offline` and `--dry-run`. A genuine *no-result* source failure should fall back to previously cached lawful metadata or generic tropes and label fallback; it must not fabricate news/trending stats.
10. Add a minimal UI selection after headless tests: topic/genre/tone inputs, auto-pick vs human review, candidate previews with **text and source links**, and provider opt-ins. Do not make API access mandatory.

## Example brief

```yaml
cat_factory:
  creative_brief:
    topics: ["lazy cat", "kung-fu cat", "food theft"]
    genres: ["action parody", "everyday meme"]
    tone: "deadpan, absurd, believable"
    avoid: ["sad endings", "real animal harm", "politics"]
    target_duration_s: 10
    aspect_ratio: "9:16"
    max_candidates: 10
    freshness_days: 30
  discovery:
    providers_enabled: ["fixtures"] # add permitted RSS/API provider by opt-in
    max_network_calls: 10
    enable_caching: true
```

## Example normalized candidate (dummy fixture — no claim of trend)

```json
{
  "source_id": "fixture-movie-trope-01",
  "provider": "fixtures",
  "source_kind": "classic_trope",
  "title": "Hero evades attack in slow motion",
  "source_url": null,
  "retrieved_at": null,
  "evidence_of_trend": null,
  "popularity": "unverified",
  "comic_mechanism": "an ordinary character is suddenly absurdly athletic",
  "cat_parody": "A chubby cat backflips to protect its food bowl",
  "rights_notes": "New imagery, dialogue, sound and staging required"
}
```

## Tests and acceptance

- **AC-01.1:** Topics + genres alone produce **at least five** valid original concepts from offline fixtures; no external network.
- **AC-01.2:** Actual optional provider can be enabled and disabled cleanly; query response is normalized and provenance is saved.
- **AC-01.3:** Search failure/no results/cache expiry/timeouts/rate limits are distinguished; fallback never invents a trending claim.
- **AC-01.4:** Source-injected malicious instructions and malformed URLs are inert test data.
- **AC-01.5:** Duplicates and near-literal source dialogue are rejected or flagged for review.
- **AC-01.6:** Ranking considers *shot generation feasibility* for realistic cats, not just apparent popularity.
- **AC-01.7:** Final idea is selected, validated, checkpointed **before** Step 04 begins local ComfyUI rendering (FLUX 3 keyframes, then Wan2.2 GGUF I2V at 480×832; the production-resolution 1080×1920 output path remains unverified).

## OpenCode task prompt

> Implement `01-automatic-meme-discovery.md` in the current Pixelle-Video extension after reading `00-START-HERE.md`, `03-llm-integration.md` and current status. Build the fixture provider, one verified opt-in metadata provider, normalized provenance, scoring and original-parody schema; test offline without downloading copyrighted clips. Do not implement image/video generation or posting. Run all tests and report exact code and evidence; stop at this step.
