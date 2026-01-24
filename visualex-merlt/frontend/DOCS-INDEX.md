# MERLT Plugin - Documentation Index

Indice completo della documentazione del sistema di build per il plugin MERLT standalone.

## Quick Start

```bash
# 1. Check configuration
chmod +x check-config.sh
./check-config.sh

# 2. Install dependencies
npm install

# 3. Build plugin
npm run build:plugin

# 4. Verify build
chmod +x verify-build.sh
./verify-build.sh
```

## Documentation Files

### 📋 Overview Documents

#### BUILD-SYSTEM-SUMMARY.md
**Quick reference** - Start here!
- Files created/modified
- Build commands
- Output structure
- Key features
- Integration steps
- Verification steps
- Configuration summary

**Read this first** for a quick overview of the entire build system.

---

#### README-BUILD-SYSTEM.md
**Complete guide** - In-depth reference
- File structure
- Build system components
- Build process explained
- Plugin architecture
- Platform integration
- Development workflow
- Troubleshooting
- Performance optimization

**Read this** for comprehensive understanding of the system.

---

### 🔧 Technical Documentation

#### BUILD.md
**Build system details**
- Configuration files explained
- Build commands
- Output directory structure
- Verifying builds
- Analyzing bundles
- Optimizations (code splitting, CSS, tree shaking)
- CI/CD integration
- Performance metrics

**Read this** when you need to understand or modify the build process.

---

#### PLUGIN.md
**Plugin architecture**
- Plugin design overview
- Configuration details (vite.config.ts, tsconfig, package.json)
- CSS handling (inline vs separate)
- Debugging techniques
- Dependency management
- Code splitting options
- Troubleshooting common issues

**Read this** to understand the plugin's internal architecture.

---

### 🔌 Integration Documentation

#### USAGE.md
**Platform integration guide**
- Installation (local + npm)
- Dynamic import pattern
- Plugin registry setup
- Plugin manager implementation
- Rendering slot components
- Event emission
- Plugin context setup
- TypeScript types
- Feature flags
- Testing examples

**Read this** when integrating the plugin into visualex-platform.

---

#### ../docs/PLUGIN_ARCHITECTURE.md
**MERLT Plugin Architecture** (Top-level documentation)
- System overview and diagrams
- User enablement flow
- EventBus pub/sub system
- Available plugin slots (8 slots)
- Complete event types (25+ events)
- Plugin manifest and lifecycle
- MERLT plugin implementation details
- Component breakdown (8 components)
- Feature flags integration
- Deployment checklist
- Troubleshooting guide

**Read this first** for comprehensive understanding of the MERLT plugin architecture.

---

#### platform-types-example.d.ts
**Required platform types**
- Complete type definitions
- Plugin interface
- PluginManifest interface
- PluginContext interface
- SlotComponent interface
- Event types

**Use this** as reference for types that visualex-platform must provide.

---

### 🛠️ Utility Scripts

#### check-config.sh
**Pre-build configuration checker**
- Verifies all config files exist
- Checks package.json fields
- Validates vite.config.ts
- Checks TypeScript config
- Verifies dependencies installed

**Run this** BEFORE building to catch configuration issues.

```bash
chmod +x check-config.sh
./check-config.sh
```

---

#### verify-build.sh
**Post-build verification script**
- Checks build output exists
- Verifies bundle size (<2MB)
- Tests plugin exports
- Validates React is external
- Checks package.json correctness

**Run this** AFTER building to verify output is correct.

```bash
chmod +x verify-build.sh
./verify-build.sh
```

---

## Reading Path by Use Case

### 🎯 I want to understand MERLT plugin architecture

1. **../docs/PLUGIN_ARCHITECTURE.md** - System overview and design
2. Review the 8 plugin slots and 25+ event types
3. Understand feature flags integration
4. Check deployment checklist

**This is the single source of truth for plugin architecture.**

---

### 🎯 I want to build the plugin NOW

1. **BUILD-SYSTEM-SUMMARY.md** - Quick overview
2. Run `check-config.sh` - Verify config
3. Run `npm run build:plugin` - Build
4. Run `verify-build.sh` - Verify

---

### 🧑‍💻 I'm a developer maintaining the plugin

1. **../docs/PLUGIN_ARCHITECTURE.md** - MERLT plugin specifics
2. **README-BUILD-SYSTEM.md** - Complete build understanding
3. **BUILD.md** - Build system deep dive
4. **PLUGIN.md** - Plugin architecture
5. Bookmark troubleshooting sections

---

### 🔗 I'm integrating the plugin into the platform

1. **../docs/PLUGIN_ARCHITECTURE.md** - Architecture overview
2. **USAGE.md** - Integration guide
3. **platform-types-example.d.ts** - Required types
4. **BUILD-SYSTEM-SUMMARY.md** § Integration
5. Test with local installation first

---

### 🐛 I'm debugging build issues

1. Run `check-config.sh` - Check config
2. **BUILD.md** § Troubleshooting
3. **PLUGIN.md** § Troubleshooting
4. Run `verify-build.sh` - Check output
5. Check specific error in docs

---

### ⚡ I'm optimizing bundle size

1. **BUILD.md** § Optimization
2. **README-BUILD-SYSTEM.md** § Performance Optimization
3. **PLUGIN.md** § Code Splitting
4. Use `npx vite-bundle-visualizer`

---

### 🚀 I'm setting up CI/CD

1. **BUILD.md** § CI/CD Integration
2. **README-BUILD-SYSTEM.md** § CI/CD Integration
3. Use `check-config.sh` in pipeline
4. Use `verify-build.sh` for validation

---

## File Organization

```
frontend/
│
├── Configuration
│   ├── vite.config.ts              # Vite build config (dual-mode)
│   ├── tsconfig.json               # Base TypeScript config
│   ├── tsconfig.plugin.json        # Plugin TypeScript config
│   ├── package.json                # Package metadata
│   └── .npmignore                  # npm publish exclusions
│
├── Source Code
│   └── src/
│       └── plugin/
│           └── index.ts            # Plugin entry point ⭐
│
├── Documentation
│   ├── DOCS-INDEX.md               # This file
│   ├── BUILD-SYSTEM-SUMMARY.md     # Quick overview
│   ├── README-BUILD-SYSTEM.md      # Complete guide
│   ├── BUILD.md                    # Build system details
│   ├── PLUGIN.md                   # Plugin architecture
│   ├── USAGE.md                    # Integration guide
│   └── platform-types-example.d.ts # Platform types reference
│
├── Utility Scripts
│   ├── check-config.sh             # Pre-build checker
│   └── verify-build.sh             # Post-build verifier
│
└── Build Output (generated)
    └── dist/
        ├── merlt-plugin.js         # ESM bundle
        ├── merlt-plugin.js.map     # Source maps
        └── types/
            └── plugin/
                └── index.d.ts      # Type declarations
```

## Documentation Stats

| Document | Lines | Focus | Audience |
|----------|-------|-------|----------|
| BUILD-SYSTEM-SUMMARY.md | ~400 | Quick reference | Everyone |
| README-BUILD-SYSTEM.md | ~600 | Complete guide | Developers |
| BUILD.md | ~500 | Build details | Build engineers |
| PLUGIN.md | ~400 | Architecture | Plugin developers |
| USAGE.md | ~600 | Integration | Platform developers |
| platform-types-example.d.ts | ~100 | Type reference | TypeScript users |
| check-config.sh | ~200 | Pre-build check | CI/CD |
| verify-build.sh | ~200 | Post-build check | CI/CD |

## Key Concepts

### Plugin Architecture
- **Entry Point**: `src/plugin/index.ts`
- **Export**: Default export of `Plugin` object
- **Interface**: Defined by `@visualex/platform/lib/plugins`
- **Loading**: Dynamic import by platform

### Build System
- **Tool**: Vite in library mode
- **Mode**: Dual (dev + plugin)
- **Output**: ESM single bundle
- **External**: React, ReactDOM
- **CSS**: Inlined in bundle

### Type System
- **Tool**: TypeScript compiler (tsc)
- **Config**: tsconfig.plugin.json
- **Output**: dist/types/
- **Entry**: dist/types/plugin/index.d.ts

### Integration
- **Method**: Dynamic import
- **Install**: npm install (local or registry)
- **Initialize**: Async with context
- **Cleanup**: Function returned from initialize

## Common Tasks

### Build from scratch
```bash
npm install
npm run build:plugin
```

### Verify configuration
```bash
./check-config.sh
```

### Verify build output
```bash
./verify-build.sh
```

### Test plugin export
```bash
node -e "import('./dist/merlt-plugin.js').then(m => console.log(m.default))"
```

### Analyze bundle size
```bash
npx vite-bundle-visualizer dist/merlt-plugin.js
```

### Install in platform
```bash
cd ../visualex-platform
npm install file:../visualex-merlt/frontend
```

### Type check only
```bash
npm run typecheck
```

### Build types only
```bash
npm run build:plugin:types
```

### Build bundle only
```bash
npm run build:plugin:bundle
```

## Support Resources

### For Build Issues
1. **check-config.sh** output
2. **BUILD.md** § Troubleshooting
3. **README-BUILD-SYSTEM.md** § Troubleshooting

### For Integration Issues
1. **USAGE.md** § Troubleshooting
2. **platform-types-example.d.ts**
3. **BUILD-SYSTEM-SUMMARY.md** § Platform Requirements

### For Architecture Questions
1. **PLUGIN.md**
2. **README-BUILD-SYSTEM.md** § Plugin Architecture
3. **src/plugin/index.ts** (source code)

### For Performance Optimization
1. **BUILD.md** § Optimization
2. **README-BUILD-SYSTEM.md** § Performance Optimization
3. Bundle analyzer output

## Version History

### v1.0 (Current)
- ✅ Dual-mode Vite configuration
- ✅ TypeScript type generation
- ✅ ESM output with React external
- ✅ Complete documentation suite
- ✅ Verification scripts
- ✅ CI/CD ready

## Contributing

When updating the build system:

1. Update relevant docs (BUILD.md, PLUGIN.md, etc.)
2. Update BUILD-SYSTEM-SUMMARY.md
3. Test with `check-config.sh`
4. Build with `npm run build:plugin`
5. Verify with `verify-build.sh`
6. Update this index if new docs added

## License

Same as parent project (see LICENSE in repository root).

---

**Last Updated**: 2026-01-18

**Maintained By**: VisuaLex Development Team

**Questions?** Check the troubleshooting sections in BUILD.md and PLUGIN.md first.
