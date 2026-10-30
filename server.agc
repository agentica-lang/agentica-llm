## The inference server: Bond routes in front of an engine task that batches requests on llama.cpp.
##
## agentica server.agc [--model PATH] [--port N] [--threads N] [--ctx N] [--parallel N] [--batch N]
##                     [--backends DIR]

import Api from "src/api"
import Bond from "pkg/bond/bond"
import Engines from "src/engine"
import Llama from "src/llama"
import HTTP from "std/http"

REQUEST_QUEUE = 4096

struct Options
  model: String = "deps/qwen2.5-0.5b-instruct-q4_k_m.gguf"
  backends: String = "native/linux"
  host: String = "127.0.0.1"
  port: Int = 8080
  threads: Int = 8
  n_ctx: Int = 8192
  parallel: Int = 8
  n_batch: Int = 2048
end

struct EngineJob
  model: Llama.Model
  config: Engines.EngineConfig
  requests: Channel[Engines.GenRequest]
end

def main() -> Void
  case serve(System.args())
    ok(_) -> puts("stopped")
    error(message) ->
      puts("error: #{message}")
      System.exit(1)
  end
end

def serve(args: List[String]) -> Result[Void]
  options = try parse(args, Options())
  Llama.init(options.backends)
  model = try Llama.load_model(options.model)
  requests: Channel[Engines.GenRequest] = Channel.new(REQUEST_QUEUE)
  config = Engines.EngineConfig(
    n_ctx: options.n_ctx,
    n_batch: options.n_batch,
    n_parallel: options.parallel,
    threads: options.threads,
  )
  engine = Task.spawn(&run_engine, EngineJob(model: model, config: config, requests: requests))
  api = Api.new_api(model, requests, "qwen2.5-0.5b-instruct")
  server = try Bond.start_with(
    Api.app(api),
    options.host,
    options.port,
    HTTP.server_options().with(max_connections: 1024, max_body_bytes: 4_194_304),
  )
  puts("listening on #{options.host}:#{Bond.local_port(server)}")
  IO.flush()
  try Bond.wait(server)
  return Task.await(engine)
end

def run_engine(job: EngineJob) -> Result[Void]
  return Engines.run(job.model, job.config, job.requests)
end

# `--name value` pairs; a flag that is not known or has no value is an error.
def parse(args: List[String], start: Options) -> Result[Options]
  options = start
  i = 0
  while i < args.size
    if i + 1 >= args.size
      return error("#{args[i]} needs a value")
    end
    options = try apply(options, args[i], args[i + 1])
    i += 2
  end
  return ok(options)
end

def apply(options: Options, flag: String, value: String) -> Result[Options]
  case flag
    "--model" -> return ok(options.with(model: value))
    "--backends" -> return ok(options.with(backends: value))
    "--host" -> return ok(options.with(host: value))
    "--port" -> return ok(options.with(port: try value.to_i))
    "--threads" -> return ok(options.with(threads: try value.to_i))
    "--ctx" -> return ok(options.with(n_ctx: try value.to_i))
    "--parallel" -> return ok(options.with(parallel: try value.to_i))
    "--batch" -> return ok(options.with(n_batch: try value.to_i))
    _ -> return error("unknown flag #{flag}")
  end
end
