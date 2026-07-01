# Documentation guide (rlx-swift)

How documentation is organized, **how to view** compiled DocC output, and **how to document
new APIs** going forward.

| I want to… | Read |
|------------|------|
| **Extend or integrate the library** | **[DeveloperGuide.md](DeveloperGuide.md)** (start here for day-to-day development) |
| **Locked API contracts** | [`design.md`](../design.md) |
| **Browse APIs in Xcode** | Sections below (*How to view compiled documentation*) |
| **Add docs for a new public API** | *How to document new APIs / code* below |

The **normative** contract text remains [`design.md`](../design.md). DocC and these guides
**teach and navigate** the API; they do not replace locked design decisions.
Update `design.md` in the same PR as intentional contract changes.

---

## Layers at a glance

| Resource | Purpose | Committed? |
|----------|---------|------------|
| [`design.md`](../design.md) | Locked API contracts, roadmap, decisions | Yes |
| **[DeveloperGuide.md](DeveloperGuide.md)** | Module map, patterns, errors, PR checklist | Yes |
| `Sources/<Target>/<Target>.docc/` | DocC catalogs (API + articles) | Yes |
| `///` comments on public Swift APIs | Symbol pages in DocC | Yes |
| [`README.md`](../README.md) | Install, CI, package layout | Yes |
| `Examples/RandomAgentDemo` | Runnable random-policy demo | Yes |
| `.build/plugins/Swift-DocC/outputs/*.doccarchive` | Compiled DocC archives | No (generated) |
| `.build/docc-site/<Target>/` | Static HTML sites (from `./scripts/generate-docs.sh`) | No (generated) |

**Important:** The `.docc` folders under `Sources/` are **hand-authored** (markdown + layout).
The **swift-docc-plugin** only **compiles** catalogs + symbol comments into archives/HTML.
It does not invent catalogs for you.

---

## How to view compiled documentation

You need **full Xcode** so the `docc` tool is available (Command Line Tools alone are not enough).

```bash
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
```

### Option A — Xcode documentation viewer (recommended day-to-day)

1. Open the package in Xcode: **File → Open** → `Package.swift` (or the repo folder).
2. **Product → Build Documentation** (or use the documentation window).
3. Browse modules (**RLXCore**, **RLXWrappers**, …) in Xcode’s documentation viewer.
4. Search for symbols (e.g. `PassiveEnvChecker`) from the doc window.

This merges your latest `///` comments and `.docc` articles without hunting for HTML paths.

### Option B — Generate a `.doccarchive` via SwiftPM (CLI)

```bash
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer

# Example: RLXCore only
xcrun swift package generate-documentation --target RLXCore
```

Default archive location (plugin output):

```text
.build/plugins/Swift-DocC/outputs/RLXCore.doccarchive
```

**Open the archive:**

- Double-click the `.doccarchive` in Finder (opens in Xcode’s documentation viewer), **or**
- In Xcode: **File → Open** → select the `.doccarchive`.

Repeat with `--target RLXWrappers` (etc.) for other modules.

### Option C — Static HTML site (browser)

```bash
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
./scripts/generate-docs.sh
# writes → .build/docc-site/RLXCore, RLXWrappers, …
```

Then open in a browser, for example:

```bash
open .build/docc-site/RLXCore/index.html
# or
open .build/docc-site/RLXWrappers/index.html
```

Static hosting layout uses `--transform-for-static-hosting`; paths under `documentation/` in each
site root are normal for DocC static output. If a page 404s when opened as a file URL, prefer
**Option A/B** or serve the folder with a tiny HTTP server:

```bash
cd .build/docc-site/RLXCore && python3 -m http.server 8000
# visit http://localhost:8000
```

### Option D — Preview plugin (optional)

With the DocC plugin installed (already in `Package.swift`):

```bash
xcrun swift package --disable-sandbox preview-documentation --target RLXCore
```

Follow the URL printed in the terminal (local preview server), when available for your toolchain.

### If you see `Plugin does not have access to a tool named 'docc'`

Your active developer directory is almost certainly **Command Line Tools**, which has `swift`
but **not** `docc`. Fix **before** `generate-documentation`:

```bash
# 1) Point at full Xcode (required)
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
# or for this shell only:
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer

# 2) Confirm docc exists
xcrun --find docc
# expect: .../XcodeDefault.xctoolchain/usr/bin/docc

# 3) Retry with that toolchain
xcrun swift package generate-documentation --target RLXCore
```

If `xcrun --find docc` still fails, install/open **Xcode.app** from the App Store (not only CLT),
accept the license (`sudo xcodebuild -license accept`), then retry.

**Easiest path when CLI is broken:** skip the plugin and use **Option A** (Xcode → Product → Build Documentation). No `docc` on PATH required in your terminal.

---

## How to document new APIs / code (workflow)

### 1. Choose the target

| Kind of change | Document in |
|----------------|-------------|
| Protocols, spaces, registry, seed/PRNG, errors | `RLXCore` |
| Wrappers, `InfoKeys`, validation helpers | `RLXWrappers` |
| Reference / debug envs | `RLXEnvs` |
| `checkEnvironment` and test harnesses | `RLXTesting` |
| Vector envs | `RLXVector` |

### 2. Write `///` documentation on public symbols (required for symbol pages)

DocC builds symbol pages from **public** declarations and their doc comments.

```swift
/// Short summary (one sentence).
///
/// Longer discussion: behaviour, invariants, cost, when to use.
///
/// - Parameters:
///   - inner: The wrapped environment.
/// - Returns: Description if useful.
/// - Throws: ``EnvironmentError/invalidObservation(_:)`` when …
///
/// ## Example
///
/// ```swift
/// let env = PassiveEnvChecker(MyEnv())
/// ```
///
/// > Design reference: `design.md` §20.2.
public final class MyNewWrapper<Inner: Environment>: Environment, EnvironmentWrapper {
    ...
}
```

**Tips:**

- Prefer linking symbols with double backticks in DocC: ``` ``Environment`` ```, ``` ``EnvironmentError/closed`` ```.
- Document throws, defaults, thread-safety, and what the API does *not* do.
- Keep **contract** changes mirrored in `design.md` if they are normative.

### 3. Add or update a DocC article when the concept is larger than one symbol

Articles live under the target catalog:

```text
Sources/RLXWrappers/RLXWrappers.docc/
  RLXWrappers.md                 ← module landing (Topics list)
  Articles/
    ValidationLayers.md
    CustomEnvironmentGuide.md
    MyNewTopic.md                ← add when needed
```

Minimal article:

```markdown
# My new topic

## Overview

Explain the concept, with links to symbols: ``MyNewWrapper``.

### Related design sections

`design.md` §…
```

Register it on the module page’s **Topics** (in `RLXWrappers.md` or the relevant root):

```markdown
## Topics

### Getting started

- <doc:ValidationLayers>
- <doc:MyNewTopic>
```

New catalogs for a **new library target**: create `Sources/NewTarget/NewTarget.docc/NewTarget.md`
with `# ``NewTarget``` and `@Metadata { @TechnologyRoot }` (see existing modules).

### 4. Rebuild and view

```bash
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
xcrun swift package generate-documentation --target RLXWrappers
# open .build/plugins/Swift-DocC/outputs/RLXWrappers.doccarchive
```

Or **Product → Build Documentation** in Xcode and confirm the new symbol/article appears.

### 5. Runnable examples (optional but valuable)

- Small snippets in `///` or articles.
- End-to-end demos as executables under `Examples/` (see `RandomAgentDemo`) when users need a
  full loop, not just an API page.

### 6. PR checklist for documentation

- [ ] Public API has accurate `///` (summary + behaviour + errors)
- [ ] If behaviour is a **contract**, `design.md` updated in the same PR
- [ ] New conceptual surface linked from the module `.docc` Topics list
- [ ] DocC build succeeds for the touched target (`generate-documentation` or Xcode)
- [ ] README / this guide only if install or doc *workflow* changed

---

## Source → compiled flow (mental model)

```text
  YOU commit:
    Sources/Foo/*.swift          (/// on public APIs)
    Sources/Foo/Foo.docc/        (landing page + Articles/*.md)
    design.md                    (normative contracts)
    Documentation/README.md      (this guide)

  PLUGIN + docc compile:
    symbol graph  +  catalog  →  .doccarchive  /  static HTML under .build/

  YOU view:
    Xcode doc viewer  |  open .doccarchive  |  browser on .build/docc-site
```

---

## Module reading order

1. **RLXCore** — lifecycle, spaces, seeding, info/specs
2. **RLXWrappers** — validation layers, composition, custom env guide
3. **RLXEnvs** / **RLXTesting** / **RLXVector** as needed

Custom env walkthrough (also in DocC):
[`Sources/RLXWrappers/RLXWrappers.docc/Articles/CustomEnvironmentGuide.md`](../Sources/RLXWrappers/RLXWrappers.docc/Articles/CustomEnvironmentGuide.md)

---

## Related design sections

| Topic | `design.md` |
|-------|-------------|
| Goals / non-goals | §1–§4 |
| Architecture | §6–§8 |
| Lifecycle & termination | §11–§12 |
| Seeding | §13 |
| Info keys | §14 |
| Wrappers | §15 |
| Vector envs | §16 |
| Validation | §20 |
| Concurrency | §21 |
| Reference envs | §24 |
| PR plan | §28 |
| Protocol sketch | Appendix A |
