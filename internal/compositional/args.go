package compositional

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
)

// parseArgs converts a Python-literal argument list into a JSON object string.
// Top-level keyword arguments (ident=value) become object keys; positional
// arguments are stored best-effort under _arg0, _arg1, ... . On any parse error
// the whole argument string is preserved under "_raw" so the call is never lost.
func parseArgs(args string) string {
	if strings.TrimSpace(args) == "" {
		return "{}"
	}
	obj, err := parseKwargs(args)
	if err != nil {
		raw, _ := json.Marshal(args)
		return `{"_raw":` + string(raw) + `}`
	}
	return obj
}

// keyRe matches a leading "ident =" assignment (not "==").
var keyRe = regexp.MustCompile(`^([A-Za-z_]\w*)\s*=`)

type argParser struct {
	s   string
	pos int
}

func parseKwargs(s string) (string, error) {
	p := &argParser{s: s}
	var parts []string
	argIdx := 0
	for {
		p.skipSpace()
		if p.pos >= len(p.s) {
			break
		}
		key, ok := p.tryKey()
		if !ok {
			key = fmt.Sprintf("_arg%d", argIdx)
			argIdx++
		}
		val, err := p.parseValue()
		if err != nil {
			return "", err
		}
		kb, _ := json.Marshal(key)
		parts = append(parts, string(kb)+":"+val)

		p.skipSpace()
		if p.pos >= len(p.s) {
			break
		}
		if p.s[p.pos] == ',' {
			p.pos++
			continue
		}
		return "", fmt.Errorf("compositional: unexpected character %q at %d", p.s[p.pos], p.pos)
	}
	if len(parts) == 0 {
		return "{}", nil
	}
	return "{" + strings.Join(parts, ",") + "}", nil
}

func (p *argParser) tryKey() (string, bool) {
	m := keyRe.FindStringSubmatchIndex(p.s[p.pos:])
	if m == nil {
		return "", false
	}
	matchEnd := p.pos + m[1] // index just past '='
	if matchEnd < len(p.s) && p.s[matchEnd] == '=' {
		// This is "==", a comparison, not a keyword assignment.
		return "", false
	}
	key := p.s[p.pos+m[2] : p.pos+m[3]]
	p.pos = matchEnd
	return key, true
}

// parseValue parses a single Python literal and returns its JSON encoding.
func (p *argParser) parseValue() (string, error) {
	p.skipSpace()
	if p.pos >= len(p.s) {
		return "", fmt.Errorf("compositional: unexpected end of arguments")
	}
	switch c := p.s[p.pos]; c {
	case '\'', '"':
		str, err := p.parseString()
		if err != nil {
			return "", err
		}
		b, _ := json.Marshal(str)
		return string(b), nil
	case '[':
		return p.parseList()
	case '{':
		return p.parseDict()
	default:
		return p.parseAtom()
	}
}

func (p *argParser) parseString() (string, error) {
	quote := p.s[p.pos]
	p.pos++
	var b strings.Builder
	for p.pos < len(p.s) {
		c := p.s[p.pos]
		if c == '\\' {
			p.pos++
			if p.pos >= len(p.s) {
				return "", fmt.Errorf("compositional: dangling escape in string")
			}
			switch e := p.s[p.pos]; e {
			case 'n':
				b.WriteByte('\n')
			case 't':
				b.WriteByte('\t')
			case 'r':
				b.WriteByte('\r')
			case '\\':
				b.WriteByte('\\')
			case '\'':
				b.WriteByte('\'')
			case '"':
				b.WriteByte('"')
			default:
				b.WriteByte(e)
			}
			p.pos++
			continue
		}
		if c == quote {
			p.pos++
			return b.String(), nil
		}
		b.WriteByte(c)
		p.pos++
	}
	return "", fmt.Errorf("compositional: unterminated string literal")
}

func (p *argParser) parseList() (string, error) {
	p.pos++ // consume '['
	var items []string
	for {
		p.skipSpace()
		if p.pos >= len(p.s) {
			return "", fmt.Errorf("compositional: unterminated list")
		}
		if p.s[p.pos] == ']' {
			p.pos++
			break
		}
		v, err := p.parseValue()
		if err != nil {
			return "", err
		}
		items = append(items, v)
		p.skipSpace()
		if p.pos >= len(p.s) {
			return "", fmt.Errorf("compositional: unterminated list")
		}
		switch p.s[p.pos] {
		case ',':
			p.pos++
		case ']':
			p.pos++
			return "[" + strings.Join(items, ",") + "]", nil
		default:
			return "", fmt.Errorf("compositional: unexpected character %q in list", p.s[p.pos])
		}
	}
	return "[" + strings.Join(items, ",") + "]", nil
}

func (p *argParser) parseDict() (string, error) {
	p.pos++ // consume '{'
	var parts []string
	for {
		p.skipSpace()
		if p.pos >= len(p.s) {
			return "", fmt.Errorf("compositional: unterminated dict")
		}
		if p.s[p.pos] == '}' {
			p.pos++
			break
		}
		key, err := p.parseDictKey()
		if err != nil {
			return "", err
		}
		p.skipSpace()
		if p.pos >= len(p.s) || p.s[p.pos] != ':' {
			return "", fmt.Errorf("compositional: expected ':' in dict")
		}
		p.pos++
		val, err := p.parseValue()
		if err != nil {
			return "", err
		}
		kb, _ := json.Marshal(key)
		parts = append(parts, string(kb)+":"+val)
		p.skipSpace()
		if p.pos >= len(p.s) {
			return "", fmt.Errorf("compositional: unterminated dict")
		}
		switch p.s[p.pos] {
		case ',':
			p.pos++
		case '}':
			p.pos++
			return "{" + strings.Join(parts, ",") + "}", nil
		default:
			return "", fmt.Errorf("compositional: unexpected character %q in dict", p.s[p.pos])
		}
	}
	return "{" + strings.Join(parts, ",") + "}", nil
}

// parseDictKey reads a dict key (string literal or bareword/number) and returns
// it coerced to a string.
func (p *argParser) parseDictKey() (string, error) {
	p.skipSpace()
	if p.pos >= len(p.s) {
		return "", fmt.Errorf("compositional: unterminated dict key")
	}
	if c := p.s[p.pos]; c == '\'' || c == '"' {
		return p.parseString()
	}
	start := p.pos
	for p.pos < len(p.s) && p.s[p.pos] != ':' && !isSpace(p.s[p.pos]) && p.s[p.pos] != ',' && p.s[p.pos] != '}' {
		p.pos++
	}
	key := strings.TrimSpace(p.s[start:p.pos])
	if key == "" {
		return "", fmt.Errorf("compositional: empty dict key")
	}
	return key, nil
}

var (
	intRe   = regexp.MustCompile(`^-?\d+$`)
	floatRe = regexp.MustCompile(`^-?\d*\.\d+$`)
)

func (p *argParser) parseAtom() (string, error) {
	start := p.pos
	for p.pos < len(p.s) && !isAtomDelim(p.s[p.pos]) {
		p.pos++
	}
	token := strings.TrimSpace(p.s[start:p.pos])
	if token == "" {
		return "", fmt.Errorf("compositional: empty value")
	}
	switch token {
	case "True":
		return "true", nil
	case "False":
		return "false", nil
	case "None":
		return "null", nil
	}
	if intRe.MatchString(token) || floatRe.MatchString(token) {
		return token, nil
	}
	b, _ := json.Marshal(token)
	return string(b), nil
}

func isAtomDelim(c byte) bool {
	return c == ',' || c == ')' || c == ']' || c == '}' || isSpace(c)
}

func (p *argParser) skipSpace() {
	for p.pos < len(p.s) && isSpace(p.s[p.pos]) {
		p.pos++
	}
}
