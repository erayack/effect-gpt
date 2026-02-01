/**
 * Seeded model factories for reproducible tests.
 * All factories accept optional seed to produce deterministic weights.
 */
import { seeded, type Rng } from "../../../src/tensor/random"
import { CANONICAL_SEED } from "./seed"
import { EMBEDDING_DIM, HIDDEN_DIM, MAX_SEQ_LEN } from "../../../src/config"
import { Embeddings } from "../../../src/model/Embeddings"
import { SelfAttention } from "../../../src/model/SelfAttention"
import { FeedForward } from "../../../src/model/FeedForward"
import { LayerNorm } from "../../../src/model/LayerNorm"
import { TransformerBlock } from "../../../src/model/TransformerBlock"
import { OutputProjection } from "../../../src/model/OutputProjection"
import { LLM } from "../../../src/model/LLM"
import { Vocab } from "../../../src/vocab/Vocab"
import type { ModelLayer } from "../../../src/model/ModelLayer"

export interface EmbeddingsOptions {
  seed?: number
  embeddingDim?: number
  maxSeqLen?: number
}

export const makeEmbeddings = (vocabSize: number, options: EmbeddingsOptions = {}): Embeddings => {
  const { seed = CANONICAL_SEED, embeddingDim = EMBEDDING_DIM, maxSeqLen = MAX_SEQ_LEN } = options
  const rng = seeded(seed)
  return new Embeddings(vocabSize, embeddingDim, maxSeqLen, rng)
}

export interface SelfAttentionOptions {
  seed?: number
  embeddingDim?: number
}

export const makeSelfAttention = (options: SelfAttentionOptions = {}): SelfAttention => {
  const { seed = CANONICAL_SEED, embeddingDim = EMBEDDING_DIM } = options
  const rng = seeded(seed)
  return new SelfAttention(embeddingDim, rng)
}

export interface FeedForwardOptions {
  seed?: number
  embeddingDim?: number
  hiddenDim?: number
}

export const makeFeedForward = (options: FeedForwardOptions = {}): FeedForward => {
  const { seed = CANONICAL_SEED, embeddingDim = EMBEDDING_DIM, hiddenDim = HIDDEN_DIM } = options
  const rng = seeded(seed)
  return new FeedForward(embeddingDim, hiddenDim, rng)
}

export interface LayerNormOptions {
  embeddingDim?: number
}

export const makeLayerNorm = (options: LayerNormOptions = {}): LayerNorm => {
  const { embeddingDim = EMBEDDING_DIM } = options
  return new LayerNorm(embeddingDim)
}

export interface TransformerBlockOptions {
  seed?: number
  embeddingDim?: number
  hiddenDim?: number
}

export const makeTransformerBlock = (options: TransformerBlockOptions = {}): TransformerBlock => {
  const { seed = CANONICAL_SEED, embeddingDim = EMBEDDING_DIM, hiddenDim = HIDDEN_DIM } = options
  const rng = seeded(seed)
  return new TransformerBlock(embeddingDim, hiddenDim, rng)
}

export interface OutputProjectionOptions {
  seed?: number
  embeddingDim?: number
}

export const makeOutputProjection = (
  vocabSize: number,
  options: OutputProjectionOptions = {}
): OutputProjection => {
  const { seed = CANONICAL_SEED, embeddingDim = EMBEDDING_DIM } = options
  const rng = seeded(seed)
  return new OutputProjection(embeddingDim, vocabSize, rng)
}

/**
 * Creates a fresh RNG instance for custom use in tests.
 */
export const makeRng = (seed: number = CANONICAL_SEED): Rng => seeded(seed)

export interface LLMOptions {
  seed?: number
  vocabWords?: ReadonlyArray<string>
  numTransformerBlocks?: number
}

export const makeLLM = (options: LLMOptions = {}): LLM => {
  const {
    seed = CANONICAL_SEED,
    vocabWords = Vocab.defaultWords(),
    numTransformerBlocks = 1
  } = options
  const vocab = Vocab.make(vocabWords)
  const rng = seeded(seed)
  const network: Array<ModelLayer> = [new Embeddings(vocab.words.length, EMBEDDING_DIM, MAX_SEQ_LEN, rng)]

  for (let i = 0; i < numTransformerBlocks; i++) {
    network.push(new TransformerBlock(EMBEDDING_DIM, HIDDEN_DIM, rng))
  }

  network.push(new OutputProjection(EMBEDDING_DIM, vocab.words.length, rng))
  return new LLM(vocab, network)
}

export interface LLMWithCustomNetworkOptions {
  vocabWords?: ReadonlyArray<string>
  network: ReadonlyArray<ModelLayer>
}

export const makeLLMWithNetwork = (options: LLMWithCustomNetworkOptions): LLM => {
  const { vocabWords = Vocab.defaultWords(), network } = options
  const vocab = Vocab.make(vocabWords)
  return new LLM(vocab, network)
}
