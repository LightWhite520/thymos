# Automatic Windows CUDA Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Thymos automatically select and expose a compatible DJL CUDA runtime on Windows without requiring a manually constructed `PATH`.

**Architecture:** Add a pure CUDA-runtime resolver plus a thin, idempotent Windows process bootstrap that runs before DJL initializes PyTorch. Preserve explicit configuration and existing CPU/non-Windows behavior, then pin OpenEden to the verified Thymos revision.

**Tech Stack:** Kotlin/JVM 21, DJL 0.34.0, PyTorch 2.7.1, JNA 5.17.0, Kotlin Test, Gradle

---

### Task 1: Pure CUDA Runtime Selection

**Files:**
- Create: `src/main/kotlin/io/thymos/DjlCudaRuntimeSelection.kt`
- Create: `src/test/kotlin/io/thymos/DjlCudaRuntimeSelectionTest.kt`

- [ ] **Step 1: Write failing selection tests**

Add tests which pass representative DJL index entries and assert that `cu128`
is selected for driver capability `130`, `cu121` is selected for capability
`124`, exact explicit `cu118` remains `cu118`, and no compatible entry returns
`null`.

- [ ] **Step 2: Run the focused test and verify failure**

Run:

```powershell
$env:JAVA_HOME='F:\SDK\JDK21'
.\gradlew.bat test --tests io.thymos.DjlCudaRuntimeSelectionTest --no-daemon --console=plain
```

Expected: compilation fails because `DjlCudaRuntimeSelection` does not exist.

- [ ] **Step 3: Implement the minimal pure resolver**

Create an internal value type carrying `pytorchVersion`, `classifier`, and
`flavor`, plus a resolver that matches only index records shaped like
`cuNNN/<classifier>/native/lib/torch.dll.gz`. Parse the requested flavor,
discard CUDA versions newer than it, and select the maximum remaining version.
Reject malformed explicit flavor values with a clear `IllegalArgumentException`.

- [ ] **Step 4: Run the focused test and verify success**

Run the command from Step 2. Expected: all
`DjlCudaRuntimeSelectionTest` tests pass.

- [ ] **Step 5: Commit the pure selection logic**

```powershell
git add src/main/kotlin/io/thymos/DjlCudaRuntimeSelection.kt src/test/kotlin/io/thymos/DjlCudaRuntimeSelectionTest.kt
git commit -m "Add DJL CUDA runtime selection"
```

### Task 2: Windows CUDA Driver And Native Path Boundary

**Files:**
- Modify: `build.gradle.kts`
- Create: `src/main/kotlin/io/thymos/WindowsCudaNativeAccess.kt`
- Create: `src/test/kotlin/io/thymos/WindowsCudaNativeAccessTest.kt`

- [ ] **Step 1: Write failing native-boundary tests**

Test the pure conversion from CUDA driver API values (`12080`, `12010`) to DJL
flavor numbers (`128`, `121`). Test that failed `cuInit` and
`cuDriverGetVersion` calls return no detected capability. Keep tests behind
injected native-call functions so they do not require NVIDIA hardware.

- [ ] **Step 2: Run the focused test and verify failure**

```powershell
$env:JAVA_HOME='F:\SDK\JDK21'
.\gradlew.bat test --tests io.thymos.WindowsCudaNativeAccessTest --no-daemon --console=plain
```

Expected: compilation fails because `WindowsCudaNativeAccess` does not exist.

- [ ] **Step 3: Add the explicit JNA dependency and native adapter**

Add `implementation("net.java.dev.jna:jna:5.17.0")`. Implement a small JNA
interface for `nvcuda.dll` (`cuInit`, `cuDriverGetVersion`) and a `kernel32`
interface for process environment mutation. The adapter must preserve the
existing process `PATH`, avoid duplicate directory entries, and throw when
Windows rejects the update.

- [ ] **Step 4: Run the focused test and verify success**

Run the command from Step 2. Expected: all
`WindowsCudaNativeAccessTest` tests pass.

- [ ] **Step 5: Commit the Windows native boundary**

```powershell
git add build.gradle.kts src/main/kotlin/io/thymos/WindowsCudaNativeAccess.kt src/test/kotlin/io/thymos/WindowsCudaNativeAccessTest.kt
git commit -m "Add Windows CUDA native bootstrap boundary"
```

### Task 3: Idempotent DJL Bootstrap

**Files:**
- Create: `src/main/kotlin/io/thymos/DjlCudaRuntimeBootstrap.kt`
- Create: `src/test/kotlin/io/thymos/DjlCudaRuntimeBootstrapTest.kt`

- [ ] **Step 1: Write failing bootstrap tests**

Using injected platform, driver, index, cache, property, and native-path
functions, prove that CPU and non-Windows calls do nothing; explicit
`PYTORCH_FLAVOR` bypasses driver detection; automatic mode selects the highest
compatible flavor; strict GPU mode reports a missing driver; and two calls
perform process mutation once.

- [ ] **Step 2: Run the focused test and verify failure**

```powershell
$env:JAVA_HOME='F:\SDK\JDK21'
.\gradlew.bat test --tests io.thymos.DjlCudaRuntimeBootstrapTest --no-daemon --console=plain
```

Expected: compilation fails because `DjlCudaRuntimeBootstrap` does not exist.

- [ ] **Step 3: Implement bootstrap orchestration**

Resolve PyTorch version and classifier through DJL `Platform`, resolve the
engine cache through DJL `Utils`, load or fetch the official version index,
create the selected runtime directory, set the effective system property, and
register that directory through `WindowsCudaNativeAccess`. Guard successful
initialization with synchronization and log one concise selection line.

- [ ] **Step 4: Run bootstrap and complete unit tests**

```powershell
$env:JAVA_HOME='F:\SDK\JDK21'
.\gradlew.bat test --no-daemon --console=plain
```

Expected: all Thymos unit tests pass.

- [ ] **Step 5: Commit bootstrap orchestration**

```powershell
git add src/main/kotlin/io/thymos/DjlCudaRuntimeBootstrap.kt src/test/kotlin/io/thymos/DjlCudaRuntimeBootstrapTest.kt
git commit -m "Bootstrap DJL CUDA runtime automatically"
```

### Task 4: Initialize CUDA Before DJL Engine Resolution

**Files:**
- Modify: `src/main/kotlin/io/thymos/DjlQwenTextAffectPredictor.kt`
- Modify: `src/test/kotlin/io/thymos/DjlQwenTextAffectPredictorTest.kt`

- [ ] **Step 1: Write the failing integration-boundary test**

Inject a bootstrap callback into the internal bundle-loading path and record
events from that callback and device-candidate resolution. Assert that bootstrap
is invoked first and receives the active device policy.

- [ ] **Step 2: Run the focused test and verify failure**

```powershell
$env:JAVA_HOME='F:\SDK\JDK21'
.\gradlew.bat test --tests io.thymos.DjlQwenTextAffectPredictorTest --no-daemon --console=plain
```

Expected: the ordering assertion fails because bundle loading does not invoke
the CUDA bootstrap.

- [ ] **Step 3: Invoke bootstrap before device candidates**

Call `DjlCudaRuntimeBootstrap.prepare(devicePolicy)` after bundle metadata
validation and immediately before `devicePolicy.candidates(engineName)`. Do not
change prediction, translator, metadata schema, or fallback behavior.

- [ ] **Step 4: Run all Thymos tests**

```powershell
$env:JAVA_HOME='F:\SDK\JDK21'
.\gradlew.bat test --no-daemon --console=plain
```

Expected: all tests pass.

- [ ] **Step 5: Commit predictor integration**

```powershell
git add src/main/kotlin/io/thymos/DjlQwenTextAffectPredictor.kt src/test/kotlin/io/thymos/DjlQwenTextAffectPredictorTest.kt
git commit -m "Prepare CUDA before Thymos engine loading"
```

### Task 5: Real GPU Verification And OpenEden Pin

**Files:**
- Modify: `D:/Project/openeden/core/build.gradle.kts`
- Modify only if needed for assertions: `D:/Project/openeden/core/src/jvmTest/kotlin/io/openeden/relationship/DjlTextAffectAnalyzerTest.kt`

- [ ] **Step 1: Verify Thymos in a fresh JVM without manual PATH changes**

Start from a PowerShell process where the DJL CUDA cache directory is not in
`PATH`, set only `THYMOS_MODEL_PATH` and `OPENEDEN_THYMOS_DEVICE=gpu`, and run the
trained-bundle test:

```powershell
$env:JAVA_HOME='F:\SDK\JDK21'
$env:OPENEDEN_THYMOS_DEVICE='gpu'
$env:THYMOS_MODEL_PATH='D:\Project\openeden\data\models\thymos-6d'
Remove-Item Env:PYTORCH_FLAVOR -ErrorAction SilentlyContinue
.\gradlew.bat test --tests "io.thymos.DjlQwenTextAffectPredictorTest.trained bundle loads and returns six finite values" --rerun-tasks --no-daemon --console=plain
```

Expected: the test passes and the runtime reports a GPU device.

- [ ] **Step 2: Push Thymos and obtain the immutable JitPack revision**

```powershell
git push origin HEAD
git rev-parse HEAD
```

Expected: push succeeds and prints the commit used for the OpenEden dependency.

- [ ] **Step 3: Update only the OpenEden Thymos dependency pin**

Replace the existing `com.github.LightWhite520:thymos:<revision>` value in
`D:/Project/openeden/core/build.gradle.kts` with the revision printed in Step 2.
Do not alter OpenEden model paths, Persona data, vector dimensions, or VQ-VAE
configuration.

- [ ] **Step 4: Verify OpenEden's affected tests without manual PATH changes**

```powershell
Set-Location D:\Project\openeden
$env:JAVA_HOME='F:\SDK\JDK21'
$env:OPENEDEN_THYMOS_DEVICE='gpu'
Remove-Item Env:PYTORCH_FLAVOR -ErrorAction SilentlyContinue
.\gradlew.bat :core:jvmTest --tests io.openeden.relationship.DjlTextAffectAnalyzerTest --tests io.openeden.runtime.pipeline.MessagePipelineTest --rerun-tasks --no-daemon --console=plain
```

Expected: both test classes pass and the inference description reports
`device=gpu(0)`.

- [ ] **Step 5: Review the OpenEden diff and commit only task-owned changes**

```powershell
git diff -- core/build.gradle.kts core/src/jvmTest/kotlin/io/openeden/relationship/DjlTextAffectAnalyzerTest.kt
git add core/build.gradle.kts core/src/jvmTest/kotlin/io/openeden/relationship/DjlTextAffectAnalyzerTest.kt
git commit -m "Use automatic Thymos CUDA bootstrap"
```

Before committing, omit the test file from `git add` if no test edit was needed.
Do not stage unrelated dirty OpenEden files.
