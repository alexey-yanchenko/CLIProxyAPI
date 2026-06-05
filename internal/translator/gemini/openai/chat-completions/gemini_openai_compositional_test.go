package chat_completions

import (
	"context"
	"strings"
	"testing"

	"github.com/tidwall/gjson"
)

const chatReqWithTool = `{"model":"gpt-5","tools":[{"type":"function","function":{"name":"append_to_scratchpad"}}]}`

func TestConvertGeminiChat_RecoversCompositionalCallStreaming(t *testing.T) {
	in := []string{
		`{"candidates":[{"index":0,"content":{"role":"model","parts":[{"text":"default_api.append_to_scratchpad(text=\"he"}]}}],"responseId":"r1","modelVersion":"m"}`,
		`{"candidates":[{"index":0,"content":{"role":"model","parts":[{"text":"llo\")"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2},"responseId":"r1","modelVersion":"m"}`,
	}

	var param any
	var out [][]byte
	for _, line := range in {
		out = append(out, ConvertGeminiResponseToOpenAI(context.Background(), "m", []byte(chatReqWithTool), nil, []byte(line), &param)...)
	}

	var name, args, content string
	for _, chunk := range out {
		tc := gjson.GetBytes(chunk, "choices.0.delta.tool_calls.0")
		if tc.Exists() {
			name = tc.Get("function.name").String()
			args = tc.Get("function.arguments").String()
		}
		if c := gjson.GetBytes(chunk, "choices.0.delta.content"); c.Type == gjson.String {
			content += c.String()
		}
	}
	if name != "append_to_scratchpad" || args != `{"text":"hello"}` {
		t.Fatalf("name=%q args=%q", name, args)
	}
	if strings.Contains(content, "default_api") {
		t.Fatalf("raw tool_code leaked into content: %q", content)
	}
}

func TestConvertGeminiChatNonStream_RecoversCompositionalCall(t *testing.T) {
	raw := []byte(`{"candidates":[{"index":0,"content":{"role":"model","parts":[{"text":"print(default_api.append_to_scratchpad(text=\"hi\"))"}]},"finishReason":"STOP"}],"responseId":"r3","modelVersion":"m"}`)

	var param any
	got := ConvertGeminiResponseToOpenAINonStream(context.Background(), "m", []byte(chatReqWithTool), nil, raw, &param)

	tc := gjson.GetBytes(got, "choices.0.message.tool_calls.0")
	if tc.Get("function.name").String() != "append_to_scratchpad" {
		t.Fatalf("tool call = %s", gjson.GetBytes(got, "choices.0.message").Raw)
	}
	if tc.Get("function.arguments").String() != `{"text":"hi"}` {
		t.Fatalf("args = %q", tc.Get("function.arguments").String())
	}
	if gjson.GetBytes(got, "choices.0.finish_reason").String() != "tool_calls" {
		t.Fatalf("finish_reason = %q", gjson.GetBytes(got, "choices.0.finish_reason").String())
	}
}
