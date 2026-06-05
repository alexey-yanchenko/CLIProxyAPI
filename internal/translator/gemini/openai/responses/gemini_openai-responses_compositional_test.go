package responses

import (
	"context"
	"strings"
	"testing"

	"github.com/tidwall/gjson"
)

// originalReqWithTool declares a function tool so the compositional allowlist is populated.
const originalReqWithTool = `{"model":"gpt-5","tools":[{"type":"function","name":"append_to_scratchpad"}]}`

func TestConvertGeminiResponses_RecoversCompositionalCallStreaming(t *testing.T) {
	// Gemini emits the tool call as text (tool_code) split across chunks, then STOP.
	in := []string{
		`data: {"candidates":[{"content":{"role":"model","parts":[{"text":"default_api.append_to_scratchpad(text=\"he"}]}}],"responseId":"r1","modelVersion":"m"}`,
		`data: {"candidates":[{"content":{"role":"model","parts":[{"text":"llo\")"}]}}],"responseId":"r1","modelVersion":"m"}`,
		`data: {"candidates":[{"content":{"role":"model","parts":[{"text":""}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2},"responseId":"r1","modelVersion":"m"}`,
	}

	var param any
	var out [][]byte
	for _, line := range in {
		out = append(out, ConvertGeminiResponseToOpenAIResponses(context.Background(), "m", []byte(originalReqWithTool), nil, []byte(line), &param)...)
	}

	var (
		gotFuncDone bool
		funcName    string
		funcArgs    string
		sawRawText  bool
	)
	for _, chunk := range out {
		ev, data := parseSSEEvent(t, chunk)
		switch ev {
		case "response.function_call_arguments.done":
			gotFuncDone = true
			funcArgs = data.Get("arguments").String()
		case "response.output_item.added":
			if data.Get("item.type").String() == "function_call" {
				funcName = data.Get("item.name").String()
			}
		case "response.output_text.delta":
			if strings.Contains(data.Get("delta").String(), "default_api") {
				sawRawText = true
			}
		}
	}

	if !gotFuncDone {
		t.Fatalf("expected a recovered function_call, got none")
	}
	if funcName != "append_to_scratchpad" {
		t.Fatalf("func name = %q", funcName)
	}
	if funcArgs != `{"text":"hello"}` {
		t.Fatalf("func args = %q", funcArgs)
	}
	if sawRawText {
		t.Fatalf("raw tool_code text leaked into output_text")
	}
}

func TestConvertGeminiResponses_NoToolsKeepsTextStreaming(t *testing.T) {
	// Without declared tools, a parenthesised phrase must remain plain text.
	in := `data: {"candidates":[{"content":{"role":"model","parts":[{"text":"call helper(x=1) now"}]},"finishReason":"STOP"}],"responseId":"r2","modelVersion":"m"}`
	var param any
	out := ConvertGeminiResponseToOpenAIResponses(context.Background(), "m", []byte(`{"model":"gpt-5"}`), nil, []byte(in), &param)

	var text string
	for _, chunk := range out {
		ev, data := parseSSEEvent(t, chunk)
		if ev == "response.output_text.done" {
			text = data.Get("text").String()
		}
	}
	if text != "call helper(x=1) now" {
		t.Fatalf("text = %q", text)
	}
}

func TestConvertGeminiResponsesNonStream_RecoversCompositionalCall(t *testing.T) {
	raw := []byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":"` + "```tool_code\\ndefault_api.append_to_scratchpad(text=\\\"hi\\\")\\n```" + `"}]},"finishReason":"STOP"}],"responseId":"r3","modelVersion":"m"}`)

	var param any
	got := ConvertGeminiResponseToOpenAIResponsesNonStream(context.Background(), "m", []byte(originalReqWithTool), nil, raw, &param)

	outputs := gjson.GetBytes(got, "output")
	var fnName, fnArgs string
	outputs.ForEach(func(_, item gjson.Result) bool {
		if item.Get("type").String() == "function_call" {
			fnName = item.Get("name").String()
			fnArgs = item.Get("arguments").String()
		}
		return true
	})
	if fnName != "append_to_scratchpad" {
		t.Fatalf("fn name = %q (output=%s)", fnName, outputs.Raw)
	}
	if fnArgs != `{"text":"hi"}` {
		t.Fatalf("fn args = %q", fnArgs)
	}
}
