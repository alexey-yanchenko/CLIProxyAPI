package imageutil

import (
	"encoding/base64"
	"encoding/binary"
)

const tokensPerTile = 258
const tileSize = 768

// CountInlineDataImageTokens estimates the Gemini token cost for a base64-encoded image.
// Gemini bills ceil(width/768) * ceil(height/768) * 258 tokens per image.
func CountInlineDataImageTokens(b64data, mimeType string) int64 {
	data, err := base64.StdEncoding.DecodeString(b64data)
	if err != nil {
		return 0
	}
	w, h := imageDimensions(data, mimeType)
	if w == 0 || h == 0 {
		return 0
	}
	tilesW := int64((w + tileSize - 1) / tileSize)
	tilesH := int64((h + tileSize - 1) / tileSize)
	return tilesW * tilesH * tokensPerTile
}

func imageDimensions(data []byte, mimeType string) (int, int) {
	switch mimeType {
	case "image/png", "image/apng":
		return pngDimensions(data)
	default:
		return jpegDimensions(data)
	}
}

func pngDimensions(data []byte) (int, int) {
	if len(data) < 24 {
		return 0, 0
	}
	if data[0] != 0x89 || data[1] != 0x50 || data[2] != 0x4E || data[3] != 0x47 {
		return 0, 0
	}
	w := int(binary.BigEndian.Uint32(data[16:20]))
	h := int(binary.BigEndian.Uint32(data[20:24]))
	return w, h
}

func jpegDimensions(data []byte) (int, int) {
	if len(data) < 2 || data[0] != 0xFF || data[1] != 0xD8 {
		return 0, 0
	}
	i := 2
	for i+4 <= len(data) {
		if data[i] != 0xFF {
			break
		}
		marker := data[i+1]
		if marker == 0xD9 {
			break
		}
		if i+4 > len(data) {
			break
		}
		segLen := int(binary.BigEndian.Uint16(data[i+2 : i+4]))
		if marker >= 0xC0 && marker <= 0xCF && marker != 0xC4 && marker != 0xC8 && marker != 0xCC {
			if i+9 <= len(data) {
				h := int(binary.BigEndian.Uint16(data[i+5 : i+7]))
				w := int(binary.BigEndian.Uint16(data[i+7 : i+9]))
				return w, h
			}
		}
		i += 2 + segLen
	}
	return 0, 0
}
