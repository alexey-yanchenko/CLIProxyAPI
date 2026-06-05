package compositional

import (
	"reflect"
	"testing"
)

var known = []string{"append_to_scratchpad", "request_pdf_pages", "append_to"}

func TestExtractFencedToolCode(t *testing.T) {
	text := "```tool_code\ndefault_api.append_to_scratchpad(text=\"hello\", page=3)\n```"
	calls, residual := Extract(text, known)
	if len(calls) != 1 {
		t.Fatalf("want 1 call, got %d (%+v)", len(calls), calls)
	}
	if calls[0].Name != "append_to_scratchpad" {
		t.Fatalf("name = %q", calls[0].Name)
	}
	if calls[0].Arguments != `{"text":"hello","page":3}` {
		t.Fatalf("args = %s", calls[0].Arguments)
	}
	if trim(residual) != "" {
		t.Fatalf("residual = %q", residual)
	}
}

func TestExtractBareCall(t *testing.T) {
	calls, residual := Extract(`default_api.request_pdf_pages(pages=[1, 2, 5])`, known)
	if len(calls) != 1 || calls[0].Name != "request_pdf_pages" {
		t.Fatalf("calls = %+v", calls)
	}
	if calls[0].Arguments != `{"pages":[1,2,5]}` {
		t.Fatalf("args = %s", calls[0].Arguments)
	}
	if trim(residual) != "" {
		t.Fatalf("residual = %q", residual)
	}
}

func TestExtractPrintWrapper(t *testing.T) {
	calls, residual := Extract(`print(default_api.append_to_scratchpad(text="hi"))`, known)
	if len(calls) != 1 || calls[0].Arguments != `{"text":"hi"}` {
		t.Fatalf("calls = %+v", calls)
	}
	if trim(residual) != "" {
		t.Fatalf("residual = %q", residual)
	}
}

func TestExtractNoPrefix(t *testing.T) {
	calls, _ := Extract(`append_to_scratchpad(text="hi")`, known)
	if len(calls) != 1 || calls[0].Name != "append_to_scratchpad" {
		t.Fatalf("calls = %+v", calls)
	}
}

func TestExtractUnknownNameStaysText(t *testing.T) {
	calls, residual := Extract(`helper(x=1) and print(2)`, known)
	if len(calls) != 0 {
		t.Fatalf("expected no calls, got %+v", calls)
	}
	if residual != `helper(x=1) and print(2)` {
		t.Fatalf("residual = %q", residual)
	}
}

func TestExtractProseAroundCall(t *testing.T) {
	text := `Sure, let me do that. default_api.append_to_scratchpad(text="x") Done.`
	calls, residual := Extract(text, known)
	if len(calls) != 1 {
		t.Fatalf("calls = %+v", calls)
	}
	if residual != `Sure, let me do that.  Done.` {
		t.Fatalf("residual = %q", residual)
	}
}

func TestExtractLiteralsAndNesting(t *testing.T) {
	calls, _ := Extract(`append_to_scratchpad(flag=True, off=False, n=None, items=[1, {"k": "v"}], pi=-3.14)`, known)
	if len(calls) != 1 {
		t.Fatalf("calls = %+v", calls)
	}
	want := `{"flag":true,"off":false,"n":null,"items":[1,{"k":"v"}],"pi":-3.14}`
	if calls[0].Arguments != want {
		t.Fatalf("args = %s", calls[0].Arguments)
	}
}

func TestExtractParenInsideString(t *testing.T) {
	calls, _ := Extract(`append_to_scratchpad(text="a (b), c")`, known)
	if len(calls) != 1 || calls[0].Arguments != `{"text":"a (b), c"}` {
		t.Fatalf("calls = %+v", calls)
	}
}

func TestExtractParallelSameTool(t *testing.T) {
	text := `append_to_scratchpad(text="a") append_to_scratchpad(text="b")`
	calls, _ := Extract(text, known)
	if len(calls) != 2 {
		t.Fatalf("want 2 calls, got %d (%+v)", len(calls), calls)
	}
	if calls[0].Arguments != `{"text":"a"}` || calls[1].Arguments != `{"text":"b"}` {
		t.Fatalf("args = %+v", calls)
	}
}

func TestExtractEmptyKnownNames(t *testing.T) {
	text := `default_api.append_to_scratchpad(text="hi")`
	calls, residual := Extract(text, nil)
	if len(calls) != 0 || residual != text {
		t.Fatalf("calls=%+v residual=%q", calls, residual)
	}
}

func TestExtractLongestNameWins(t *testing.T) {
	// "append_to" is a prefix of "append_to_scratchpad"; ensure the longer wins.
	calls, _ := Extract(`append_to_scratchpad(text="x")`, known)
	if len(calls) != 1 || calls[0].Name != "append_to_scratchpad" {
		t.Fatalf("calls = %+v", calls)
	}
}

func TestParseArgsFallbackRaw(t *testing.T) {
	// Unbalanced/garbage that the arg parser cannot handle should fall back to _raw.
	got := parseArgs(`text="unterminated`)
	if got != `{"_raw":"text=\"unterminated"}` {
		t.Fatalf("got %s", got)
	}
}

func TestParseArgsPositional(t *testing.T) {
	got := parseArgs(`"hello", 5`)
	if got != `{"_arg0":"hello","_arg1":5}` {
		t.Fatalf("got %s", got)
	}
}

// --- streaming ---

func TestProcessHoldsIncompleteCall(t *testing.T) {
	calls, residual, tail := Process(`append_to_scratchpad(text="he`, known, false)
	if len(calls) != 0 {
		t.Fatalf("calls = %+v", calls)
	}
	if trim(residual) != "" {
		t.Fatalf("residual = %q", residual)
	}
	if tail != `append_to_scratchpad(text="he` {
		t.Fatalf("tail = %q", tail)
	}
}

func TestProcessCompletesAcrossChunks(t *testing.T) {
	buf := ""
	chunks := []string{`append_to_scratchpad(text="he`, `llo")`}
	var allCalls []Call
	var allResidual string
	for _, ch := range chunks {
		buf += ch
		var calls []Call
		var residual string
		calls, residual, buf = Process(buf, known, false)
		allCalls = append(allCalls, calls...)
		allResidual += residual
	}
	// final flush
	calls, residual, _ := Process(buf, known, true)
	allCalls = append(allCalls, calls...)
	allResidual += residual

	if len(allCalls) != 1 || allCalls[0].Arguments != `{"text":"hello"}` {
		t.Fatalf("calls = %+v", allCalls)
	}
	if trim(allResidual) != "" {
		t.Fatalf("residual = %q", allResidual)
	}
}

func TestProcessTrailingPartialOpener(t *testing.T) {
	_, residual, tail := Process(`hello default_ap`, known, false)
	if residual != "hello " {
		t.Fatalf("residual = %q", residual)
	}
	if tail != "default_ap" {
		t.Fatalf("tail = %q", tail)
	}
}

func TestProcessFinalFlushesIncompleteAsText(t *testing.T) {
	calls, residual, tail := Process(`append_to_scratchpad(text="he`, known, true)
	if len(calls) != 0 {
		t.Fatalf("calls = %+v", calls)
	}
	if residual != `append_to_scratchpad(text="he` {
		t.Fatalf("residual = %q", residual)
	}
	if tail != "" {
		t.Fatalf("tail = %q", tail)
	}
}

func TestProcessPlainTextFlushes(t *testing.T) {
	_, residual, tail := Process(`just some normal words`, known, false)
	if residual != `just some normal words` {
		t.Fatalf("residual = %q", residual)
	}
	if tail != "" {
		t.Fatalf("tail = %q", tail)
	}
}

func TestProcessParallelDistinct(t *testing.T) {
	calls, _, _ := Process(`request_pdf_pages(pages=[1]) request_pdf_pages(pages=[2])`, known, true)
	if len(calls) != 2 {
		t.Fatalf("want 2, got %+v", calls)
	}
	if !reflect.DeepEqual(calls[0], Call{Name: "request_pdf_pages", Arguments: `{"pages":[1]}`}) {
		t.Fatalf("call0 = %+v", calls[0])
	}
	if !reflect.DeepEqual(calls[1], Call{Name: "request_pdf_pages", Arguments: `{"pages":[2]}`}) {
		t.Fatalf("call1 = %+v", calls[1])
	}
}

func TestMatchingParenStringAware(t *testing.T) {
	s := `(text="a (b)")`
	idx, ok := matchingParen(s, 0)
	if !ok || idx != len(s)-1 {
		t.Fatalf("idx=%d ok=%v", idx, ok)
	}
}

func trim(s string) string {
	out := ""
	for _, r := range s {
		if r != ' ' && r != '\n' && r != '\t' && r != '\r' {
			out += string(r)
		}
	}
	return out
}
