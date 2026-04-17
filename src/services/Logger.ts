import * as Effect from "effect/Effect"
import * as Context from "effect/Context"
import * as Layer from "effect/Layer"
import { Terminal } from "@effect/platform"

export type LogLevel = "debug" | "info" | "warn" | "error"

const LOG_LEVEL_PRIORITY: Record<LogLevel, number> = {
  debug: 0,
  info: 1,
  warn: 2,
  error: 3
}

export interface LoggerService {
  readonly log: (level: LogLevel, message: string, data?: Record<string, unknown>) => Effect.Effect<void>
  readonly debug: (message: string, data?: Record<string, unknown>) => Effect.Effect<void>
  readonly info: (message: string, data?: Record<string, unknown>) => Effect.Effect<void>
  readonly warn: (message: string, data?: Record<string, unknown>) => Effect.Effect<void>
  readonly error: (message: string, data?: Record<string, unknown>) => Effect.Effect<void>
}

class LoggerTag extends Context.Tag("effect-gpt/services/Logger")<LoggerTag, LoggerService>() {}

export const Logger = LoggerTag
export type LoggerServiceId = LoggerTag

type Formatter = (level: LogLevel, message: string, data?: Record<string, unknown>) => string

const formatStructured: Formatter = (level, message, data) => {
  const timestamp = new Date().toISOString()
  const dataStr = data ? ` ${JSON.stringify(data)}` : ""
  return `[${timestamp}] [${level.toUpperCase()}] ${message}${dataStr}\n`
}

const formatPretty: Formatter = (level, message, data) => {
  const prefix: Record<LogLevel, string> = { debug: "🔍", info: "ℹ️ ", warn: "⚠️ ", error: "❌" }
  const dataStr = data ? ` ${JSON.stringify(data)}` : ""
  return `${prefix[level]} ${message}${dataStr}\n`
}

const makeLogger = (
  minLevel: LogLevel,
  write: (msg: string) => Effect.Effect<void>,
  format: Formatter = formatStructured
): LoggerService => ({
  log: (level, message, data) =>
    LOG_LEVEL_PRIORITY[level] >= LOG_LEVEL_PRIORITY[minLevel]
      ? write(format(level, message, data))
      : Effect.void,
  debug: (message, data) =>
    LOG_LEVEL_PRIORITY.debug >= LOG_LEVEL_PRIORITY[minLevel]
      ? write(format("debug", message, data))
      : Effect.void,
  info: (message, data) =>
    LOG_LEVEL_PRIORITY.info >= LOG_LEVEL_PRIORITY[minLevel]
      ? write(format("info", message, data))
      : Effect.void,
  warn: (message, data) =>
    LOG_LEVEL_PRIORITY.warn >= LOG_LEVEL_PRIORITY[minLevel]
      ? write(format("warn", message, data))
      : Effect.void,
  error: (message, data) =>
    LOG_LEVEL_PRIORITY.error >= LOG_LEVEL_PRIORITY[minLevel]
      ? write(format("error", message, data))
      : Effect.void
})

export const ConsoleLoggerLive = (minLevel: LogLevel = "info"): Layer.Layer<LoggerServiceId> =>
  Layer.succeed(
    Logger,
    makeLogger(minLevel, (msg) => Effect.sync(() => process.stdout.write(msg)))
  )

export const TerminalLoggerLive = (
  minLevel: LogLevel = "info"
): Layer.Layer<LoggerServiceId, never, Terminal.Terminal> =>
  Layer.effect(
    Logger,
    Effect.gen(function* () {
      const terminal = yield* Terminal.Terminal
      return makeLogger(minLevel, (msg) =>
        terminal.display(msg).pipe(
          Effect.catchAll(() => Effect.void),
          Effect.asVoid
        )
      )
    })
  )

export const NullLoggerLive: Layer.Layer<LoggerServiceId> = Layer.succeed(
  Logger,
  makeLogger("error", () => Effect.void)
)

export const SilentLoggerLive: Layer.Layer<LoggerServiceId> = Layer.succeed(Logger, {
  log: () => Effect.void,
  debug: () => Effect.void,
  info: () => Effect.void,
  warn: () => Effect.void,
  error: () => Effect.void
})

export const PrettyLoggerLive = (minLevel: LogLevel = "info"): Layer.Layer<LoggerServiceId> =>
  Layer.succeed(
    Logger,
    makeLogger(minLevel, (msg) => Effect.sync(() => process.stdout.write(msg)), formatPretty)
  )

export const log = (level: LogLevel, message: string, data?: Record<string, unknown>) =>
  Effect.flatMap(Logger, (logger) => logger.log(level, message, data))

export const debug = (message: string, data?: Record<string, unknown>) =>
  Effect.flatMap(Logger, (logger) => logger.debug(message, data))

export const info = (message: string, data?: Record<string, unknown>) =>
  Effect.flatMap(Logger, (logger) => logger.info(message, data))

export const warn = (message: string, data?: Record<string, unknown>) =>
  Effect.flatMap(Logger, (logger) => logger.warn(message, data))

export const error = (message: string, data?: Record<string, unknown>) =>
  Effect.flatMap(Logger, (logger) => logger.error(message, data))
