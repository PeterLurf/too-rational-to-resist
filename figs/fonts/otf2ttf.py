"""Convert Inter OTF (CFF) to TTF (glyf) so matplotlib can embed it as Type 42. Inter: SIL Open Font License 1.1."""
import sys
from fontTools.ttLib import TTFont, newTable
from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.ttGlyphPen import TTGlyphPen


def convert(src, dst, max_err=1.0):
    font = TTFont(src)
    gs = font.getGlyphSet()
    glyf = newTable("glyf"); glyf.glyphs = {}; glyf.glyphOrder = font.getGlyphOrder()
    for name in font.getGlyphOrder():
        pen = TTGlyphPen(gs)
        gs[name].draw(Cu2QuPen(pen, max_err, reverse_direction=True))
        glyf[name] = pen.glyph()
    font["glyf"] = glyf
    font["loca"] = newTable("loca")
    font["maxp"].tableVersion = 0x00010000
    for k in ("maxZones", "maxTwilightPoints", "maxStorage", "maxFunctionDefs", "maxInstructionDefs",
              "maxStackElements", "maxSizeOfInstructions", "maxComponentElements"):
        setattr(font["maxp"], k, 0)
    font["maxp"].maxZones = 1
    font["head"].glyphDataFormat = 0
    font["post"].formatType = 2.0
    font["post"].extraNames = []; font["post"].mapping = {}
    del font["CFF "]
    if "VORG" in font: del font["VORG"]
    font.sfntVersion = "\x00\x01\x00\x00"
    font.save(dst)


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
