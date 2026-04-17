import * as Data from "effect/Data"
import * as Effect from "effect/Effect"
import * as Stream from "effect/Stream"
import * as Option from "effect/Option"
import * as Schema from "effect/Schema"
import { FileSystem } from "@effect/platform"

export type DatasetFormat = "json" | "csv"

export interface DatasetLoadOptions {
  readonly pretrainingPath: string
  readonly chatPath: string
  readonly format: DatasetFormat
}

export interface DatasetStreams {
  /**
   * Fresh stream producer for pretraining data.
   * Each invocation re-opens the file, so it is safe to call once per epoch.
   */
  readonly pretrainingStream: () => Stream.Stream<string, DatasetLoadError | DatasetParseError, FileSystem.FileSystem>
  /**
   * Fresh stream producer for chat training data.
   * Each invocation re-opens the file, so it is safe to call once per epoch.
   */
  readonly chatStream: () => Stream.Stream<string, DatasetLoadError | DatasetParseError, FileSystem.FileSystem>
}

export interface DatasetArrays {
  readonly pretrainingData: ReadonlyArray<string>
  readonly chatTrainingData: ReadonlyArray<string>
}

export class DatasetLoadError extends Data.TaggedError("DatasetLoadError")<{
  readonly path: string
  readonly error: unknown
}> {}

export class DatasetParseError extends Data.TaggedError("DatasetParseError")<{
  readonly path: string
  readonly error: unknown
}> {}

const TrainingItemSchema = Schema.String
const decodeTrainingItemJson = Schema.decodeUnknown(Schema.parseJson(TrainingItemSchema))

const makeFileStream = (path: string) =>
  Effect.flatMap(FileSystem.FileSystem, (fs) => Effect.succeed(fs.stream(path)))

const linesFromFile = (path: string) =>
  Stream.splitLines(
    Stream.decodeText(
      Stream.mapError(
        Stream.unwrap(makeFileStream(path)),
        (error) => new DatasetLoadError({ path, error })
      )
    )
  )

const parseJsonLine = (path: string, rawLine: string): Effect.Effect<Option.Option<string>, DatasetParseError> =>
  Effect.gen(function* () {
    const trimmed = rawLine.trim()
    if (trimmed === "[" || trimmed === "]" || trimmed.length === 0) {
      return Option.none()
    }

    const withoutComma = trimmed.endsWith(",") ? trimmed.slice(0, -1) : trimmed

    const decoded: string = yield* decodeTrainingItemJson(withoutComma).pipe(
      Effect.mapError((error) => new DatasetParseError({ path, error }))
    )

    return Option.some(decoded)
  })

const jsonStream = (path: string) =>
  Stream.mapEffect(linesFromFile(path), (line) => parseJsonLine(path, line)).pipe(
    Stream.filterMap((option) => option)
  )

const splitCsvLine = (line: string): Array<string> => {
  const fields: Array<string> = []
  let current = ""
  let inQuotes = false

  for (let i = 0; i < line.length; i++) {
    const ch = line[i]!
    if (ch === '"') {
      if (inQuotes && line[i + 1] === '"') {
        // Escaped quote
        current += '"'
        i++ // skip next
      } else {
        inQuotes = !inQuotes
      }
    } else if (ch === "," && !inQuotes) {
      fields.push(current)
      current = ""
    } else {
      current += ch
    }
  }

  if (inQuotes) {
    throw new Error("Unclosed quote in CSV line")
  }

  fields.push(current)
  return fields
}

const parseCsvLine = (path: string, rawLine: string): Effect.Effect<Option.Option<string>, DatasetParseError> =>
  Effect.gen(function* () {
    const trimmed = rawLine.trim()
    if (trimmed.length === 0) {
      return Option.none()
    }

    const joined = yield* Effect.try({
      try: () => splitCsvLine(rawLine).join(","),
      catch: (error) => new DatasetParseError({ path, error })
    })

    return Option.some(joined)
  })

const csvStream = (path: string) =>
  Stream.mapEffect(linesFromFile(path), (line) => parseCsvLine(path, line)).pipe(
    Stream.filterMap((option) => option)
  )

const streamForFormat = (path: string, format: DatasetFormat) =>
  format === "json" ? jsonStream(path) : csvStream(path)

const collectAll = (
  stream: Stream.Stream<string, DatasetLoadError | DatasetParseError, FileSystem.FileSystem>
) =>
  Stream.runCollect(stream).pipe(
    Effect.map((chunk) => Array.from(chunk))
  )

export const Dataset = {
  /**
   * Returns dataset streams that can be re-opened for each consumption.
   */
  load: (options: DatasetLoadOptions): DatasetStreams => ({
    pretrainingStream: () => streamForFormat(options.pretrainingPath, options.format),
    chatStream: () => streamForFormat(options.chatPath, options.format)
  }),

  /**
   * Convenience helper to materialize both streams into arrays.
   * Useful for small datasets or legacy call sites.
   */
  collect: (
    streams: DatasetStreams
  ): Effect.Effect<DatasetArrays, DatasetLoadError | DatasetParseError, FileSystem.FileSystem> =>
    Effect.all({
      pretrainingData: collectAll(streams.pretrainingStream()),
      chatTrainingData: collectAll(streams.chatStream())
    })
}
