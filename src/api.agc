## The OpenAI-shaped routes, on Bond: `POST /v1/chat/completions` and `GET /health`.

import Bond from "pkg/bond/bond"
import Engines from "engine"
import Llama from "llama"

DEFAULT_MAX_TOKENS = 128
ANSWER_TIMEOUT_MS = 600_000

## What the routes share: the model (to apply the chat template and tokenize), the engine's request channel and a
## table with the request counter.
export struct Api
  model: Llama.Model
  requests: Channel[Engines.GenRequest]
  counters: ATS[String, Int]
  model_name: String
end

struct ChatRequest
  messages: List[Llama.ChatMessage]
  max_tokens: Optional[Int] = none
  temperature: Optional[Float] = none
  seed: Optional[Int] = none
end

struct ReplyMessage
  role: String
  content: String
end

struct Choice
  index: Int
  message: ReplyMessage
  finish_reason: String
end

struct Usage
  prompt_tokens: Int
  completion_tokens: Int
  total_tokens: Int
end

struct ChatResponse
  id: String
  object: String
  created: Int
  model: String
  choices: List[Choice]
  usage: Usage
end

export def new_api(model: Llama.Model, requests: Channel[Engines.GenRequest], model_name: String) -> Api
  counters: ATS[String, Int] = ATS.new()
  return Api(model: model, requests: requests, counters: counters, model_name: model_name)
end

export def app(api: Api) -> Bond.App
  return (
    Bond.app()
    |> Bond.get("/health", &health)
    |> Bond.post("/v1/chat/completions") do |conn| chat(api, conn) end
  )
end

def health(conn: Bond.Conn) -> Result[Bond.Conn]
  return ok(Bond.json(conn, 200, JSON.from({"status": "ok"})))
end

def chat(api: Api, conn: Bond.Conn) -> Result[Bond.Conn]
  body = try Bond.json_body(conn, JSON.codec(ChatRequest))
  if body.messages.empty?
    return Bond.bad_request("messages must not be empty")
  end
  text = try Llama.chat_prompt(api.model, body.messages)
  prompt = try Llama.tokenize(api.model, text)
  reply: Channel[Engines.GenResult] = Channel.new(1)
  request = Engines.GenRequest(
    prompt: prompt,
    max_tokens: body.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
    temperature: body.temperature.unwrap_or(1.0),
    seed: body.seed.unwrap_or(0),
    reply: reply,
  )
  case api.requests.send(request, ANSWER_TIMEOUT_MS)
    ok(_) -> return answer(api, conn, reply)
    error(message) -> return ok(Bond.halt(conn, 503, message))
  end
end

def answer(api: Api, conn: Bond.Conn, reply: Channel[Engines.GenResult]) -> Result[Bond.Conn]
  case reply.recv(ANSWER_TIMEOUT_MS)
    ok(result) ->
      if result.finish_reason == "error"
        return ok(Bond.halt(conn, 400, result.failure))
      end
      return ok(respond(api, conn, result))
    error(message) -> return ok(Bond.halt(conn, 504, message))
  end
end

def respond(api: Api, conn: Bond.Conn, result: Engines.GenResult) -> Bond.Conn
  response = ChatResponse(
    id: "chatcmpl-#{api.counters.add("requests", 1)}",
    object: "chat.completion",
    created: Time.now_ms() / 1000,
    model: api.model_name,
    choices: [
      Choice(
        index: 0,
        message: ReplyMessage(role: "assistant", content: result.text),
        finish_reason: result.finish_reason,
      ),
    ],
    usage: Usage(
      prompt_tokens: result.prompt_tokens,
      completion_tokens: result.completion_tokens,
      total_tokens: result.prompt_tokens + result.completion_tokens,
    ),
  )
  return Bond.header(Bond.text(conn, 200, JSON.encode(response)), "content-type", "application/json")
end
