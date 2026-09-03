# Pipeline architecture

**This is the desired end state, not a description of what exists today.** Where the
repo diverges from it is listed at the bottom.

As specified:

> record real, including hold outs data -> check real data -> training runs [generate
> corpus (including synthetics) -> train model [oww, mww] ] -> generate eval synethtic
> corpus merge with hold outs -> eval trained models -> preflight check with live mic

```mermaid
flowchart TB

    subgraph SRC ["1 · Record and check real speech — host, needs a mic"]
        direction TB
        REC["Record real speech"]
        CHK["Check real data"]
        REC --> CHK
        CHK --> TRAINDATA[("Training data")]
        CHK --> HOLDDATA[("Holdout data<br/>NEVER trained on")]
    end

    subgraph RUN ["2 · Training run"]
        direction TB
        GEN["Generate corpus<br/>real + synthetic"]
        subgraph TRN ["Train model"]
            direction LR
            OWW["openWakeWord<br/>server"]
            MWW["microWakeWord<br/>ESP32 / edge"]
        end
        GEN --> OWW
        GEN --> MWW
    end

    TRAINDATA --> RUN

    subgraph EVALBOX ["3 · Eval model"]
        direction TB
        EVC["Generate eval<br/>synthetic corpus"]
        EVALSET[("Eval dataset")]
        EVAL["Evaluate trained models<br/>at matched precision, per speaker"]
        EVC --> EVALSET
        EVALSET --> EVAL
    end

    HOLDDATA -->|"merge"| EVALSET
    RUN -->|"trained models"| EVALBOX

    EVALBOX -->|"not better — change ONE thing"| RUN
    EVAL -->|"better"| PRE["4 · Preflight<br/>live mic"]
    PRE --> SHIP(["Deploy"])

    classDef store fill:#eef4ff,stroke:#4a6fa5,color:#12263f
    classDef guard fill:#fff4e6,stroke:#c47f00,color:#3d2800
    class TRAINDATA,HOLDDATA,EVALSET store
    class EVAL,PRE guard
```

The two things this shape is built around: **the holdout leaves step 1 and goes
straight into the eval dataset**, never through training; and **evaluation feeds back
into the training run**, because that loop is the pipeline — seventeen runs of it so
far.

## Two invariants the diagram depends on

**The holdout must postdate the model.** `train.py` globs `my_real_samples/`
recursively, so scoring against it reports training accuracy — it overstated detection
by ~10 points and hid a much larger gap on run-on speech before anyone noticed. That is
why step 1 produces *two* stores rather than one that gets split later, and why the
holdout edge bypasses step 2 entirely.

**The two negative wordlists must not overlap.** Training negatives and the eval corpus
the false-accept gates are scored on are generated separately and kept disjoint. A
phrase in both turns a generalisation measurement into a memorisation one.

## Where the repo diverges from this today

| step | gap |
|---|---|
| 2 · train | The microWakeWord corpus has no run-on positives and is Piper-only. On the openWakeWord side run-ons took held-out run-on detection from 5% to the 80s, and two TTS engines beat one by the largest margin since run 10. |
| 3 · evaluate | openWakeWord is not evaluated on the deployment runtime. `pyopen-wakeword` is TFLite-only and the ship candidates are `.onnx`, so they score through `openwakeword.model.Model` — comparable with `tuning.md`, not with a device. Converting them closes it. |
| 3 · evaluate | No window-alignment measurement for microWakeWord. `eval/check_model_alignment.py` is openWakeWord-only, and its framing does not transfer to a streaming detector with a sliding-window average. |
| 4 · preflight | **No microWakeWord path at all.** `test_model.py` loads through `openwakeword.model.Model`, so a microWakeWord model cannot currently be preflighted on a live mic — the last gate before deploy does not cover half the diagram. |
