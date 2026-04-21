import { runBenchmarkSuite } from "./harness"

const main = async (): Promise<void> => {
  const result = await runBenchmarkSuite({
    scenarioNames: ["train-small"],
    iterations: 5,
    warmup: 1,
    format: "text"
  })

  const training = result.scenarios[0]
  if (training === undefined) {
    throw new Error("Training scenario result missing")
  }

  console.log(`METRIC total_ms=${training.metrics.duration_ms.median.toFixed(3)}`)
  console.log(`METRIC epoch_ms=${training.metrics.epoch_ms.median.toFixed(3)}`)
  console.log(`METRIC final_loss=${training.metrics.final_loss.median.toFixed(6)}`)
  console.log(`METRIC trainable_tokens_per_sec=${training.metrics.trainable_tokens_per_sec.median.toFixed(3)}`)
  console.log(`METRIC corpus_examples=${training.metadata.corpus_examples}`)
}

await main()
