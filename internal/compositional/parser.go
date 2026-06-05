// Package compositional implements a defensive parser for Gemini "compositional"
// (a.k.a. tool_code) function calls.
//
// In functionCallingConfig.mode = AUTO, Gemini occasionally emits a tool
// invocation not as a structured functionCall part but as plain text using a
// Python-like notation, for example:
//
//	```tool_code
//	default_api.append_to_scratchpad(text="hello", page=3)
//	```
//
// or without fences / wrapped in print(...):
//
//	print(default_api.request_pdf_pages(pages=[1, 2, 5]))
//	append_to_scratchpad(text="hi")
//
// Adapters that only look for structured calls miss these and surface the text
// as ordinary output, breaking tool loops. This package recovers such calls so
// they can be re-emitted through the normal structured function-call path.
//
// Matching is restricted to an allowlist of declared tool names so that
// ordinary prose like "helper(x=1)" is never mistaken for a call.
package compositional

import (
	"regexp"
	"sort"
	"strings"
)

// Call is a recovered compositional function call.
type Call struct {
	// Name is the tool name exactly as it matched the allowlist.
	Name string
	// Arguments is a JSON object string (e.g. `{"text":"hello"}`).
	Arguments string
}

// fences are code-fence tokens stripped before scanning. Longer tokens first
// so the bare "```" does not pre-empt the more specific variants.
var fences = []string{"```tool_code", "```python", "```json", "```"}

// Extract scans text for compositional calls whose names appear in knownNames
// and returns the recovered calls together with the residual text (the input
// with the recognised calls removed). It is the non-streaming entry point and
// treats any unterminated call as ordinary text.
func Extract(text string, knownNames []string) (calls []Call, residual string) {
	calls, residual, _ = parse(text, knownNames, true)
	return calls, residual
}

// Process is the streaming entry point. In addition to calls and residual it
// returns a tail: a trailing fragment that may still grow into a call or fence
// on the next chunk. The caller must hold the tail back (not emit it) and
// prepend it to the next chunk. When final is true no tail is held back and any
// unterminated call is flushed to residual as text.
func Process(text string, knownNames []string, final bool) (calls []Call, residual string, tail string) {
	return parse(text, knownNames, final)
}

func parse(text string, knownNames []string, final bool) ([]Call, string, string) {
	if text == "" || len(knownNames) == 0 {
		return nil, text, ""
	}

	text = stripFences(text)
	re := buildOpener(knownNames)
	if re == nil {
		return nil, text, ""
	}

	var calls []Call
	var residual strings.Builder
	cursor := 0

	for cursor < len(text) {
		loc := re.FindStringSubmatchIndex(text[cursor:])
		if loc == nil {
			break
		}
		matchStart := cursor + loc[0]
		matchEnd := cursor + loc[1]
		name := text[cursor+loc[2] : cursor+loc[3]]
		openParen := matchEnd - 1 // index of the call's '('

		closeParen, ok := matchingParen(text, openParen)
		if !ok {
			// Unterminated call.
			residual.WriteString(text[cursor:matchStart])
			if final {
				residual.WriteString(text[matchStart:])
				return calls, residual.String(), ""
			}
			return calls, residual.String(), text[matchStart:]
		}

		residual.WriteString(text[cursor:matchStart])
		args := text[openParen+1 : closeParen]
		calls = append(calls, Call{Name: name, Arguments: parseArgs(args)})
		cursor = closeParen + 1

		// If the opener was a print(...) wrapper, swallow its closing ')'.
		if strings.HasPrefix(strings.TrimSpace(text[matchStart:matchEnd]), "print") {
			j := cursor
			for j < len(text) && isSpace(text[j]) {
				j++
			}
			if j < len(text) && text[j] == ')' {
				cursor = j + 1
			}
		}
	}

	rest := text[cursor:]
	if final {
		residual.WriteString(rest)
		return calls, residual.String(), ""
	}

	hold := trailingHoldback(rest, knownNames)
	residual.WriteString(rest[:len(rest)-hold])
	return calls, residual.String(), rest[len(rest)-hold:]
}

func stripFences(text string) string {
	for _, f := range fences {
		if strings.Contains(text, f) {
			text = strings.ReplaceAll(text, f, "")
		}
	}
	return text
}

// buildOpener compiles the opener regexp:
//
//	(?:print\s*\(\s*)?(?:default_api\.)?(NAME)\s*\(
//
// NAME is the alternation of the (escaped) known names, longest first so a
// longer name is preferred over a shorter prefix of it.
func buildOpener(knownNames []string) *regexp.Regexp {
	names := make([]string, 0, len(knownNames))
	seen := make(map[string]struct{}, len(knownNames))
	for _, n := range knownNames {
		if n == "" {
			continue
		}
		if _, ok := seen[n]; ok {
			continue
		}
		seen[n] = struct{}{}
		names = append(names, n)
	}
	if len(names) == 0 {
		return nil
	}
	sort.Slice(names, func(i, j int) bool { return len(names[i]) > len(names[j]) })
	for i, n := range names {
		names[i] = regexp.QuoteMeta(n)
	}
	pattern := `(?:print\s*\(\s*)?(?:default_api\.)?(` + strings.Join(names, "|") + `)\s*\(`
	re, err := regexp.Compile(pattern)
	if err != nil {
		return nil
	}
	return re
}

// matchingParen returns the index of the bracket that closes the one at
// openIdx, ignoring brackets inside string literals. ok is false when the input
// ends before the bracket is closed (i.e. the call is incomplete).
func matchingParen(s string, openIdx int) (int, bool) {
	depth := 0
	inStr := false
	var quote byte
	escaped := false
	for i := openIdx; i < len(s); i++ {
		c := s[i]
		if inStr {
			if escaped {
				escaped = false
				continue
			}
			switch c {
			case '\\':
				escaped = true
			case quote:
				inStr = false
			}
			continue
		}
		switch c {
		case '"', '\'':
			inStr = true
			quote = c
		case '(', '[', '{':
			depth++
		case ')', ']', '}':
			depth--
			if depth == 0 {
				return i, true
			}
		}
	}
	return 0, false
}

// trailingHoldback returns the length of the longest suffix of text that is a
// strict prefix of any trigger (fence, default_api., print(, name(,
// default_api.name(). That suffix may become the start of a call in the next
// chunk and must be held back.
func trailingHoldback(text string, knownNames []string) int {
	if text == "" {
		return 0
	}
	triggers := make([]string, 0, len(knownNames)*2+len(fences)+2)
	triggers = append(triggers, fences...)
	triggers = append(triggers, "default_api.", "print(")
	for _, n := range knownNames {
		if n == "" {
			continue
		}
		triggers = append(triggers, n+"(", "default_api."+n+"(")
	}

	maxHold := 0
	for _, t := range triggers {
		limit := len(t) - 1
		if limit > len(text) {
			limit = len(text)
		}
		for l := limit; l > maxHold; l-- {
			if strings.HasSuffix(text, t[:l]) {
				maxHold = l
				break
			}
		}
	}
	return maxHold
}

func isSpace(c byte) bool {
	return c == ' ' || c == '\t' || c == '\n' || c == '\r'
}
