# Draw a poster spec (JSON written by build_pptx.py) with PowerPoint, save it as .pptx and export a PDF.
# Prints one CHECK line per problem (overflow, off-slide, font below the minimum) and a MINFONT line.
#   powershell -NoProfile -ExecutionPolicy Bypass -File pptx_render.ps1 spec.json out.pptx out.pdf
param([string]$Spec, [string]$Pptx, [string]$Pdf)
$ErrorActionPreference = "Stop"

$PtPerMm = 72 / 25.4
function Pt([double]$mm) { return $mm * $PtPerMm }
function Rgb([string]$hex) {
    $h = $hex.TrimStart("#")
    $r = [Convert]::ToInt32($h.Substring(0, 2), 16)
    $g = [Convert]::ToInt32($h.Substring(2, 2), 16)
    $b = [Convert]::ToInt32($h.Substring(4, 2), 16)
    return $r + 256 * $g + 65536 * $b
}

function Set-Paragraphs($textRange2, $paras, $defaults) {
    $parts = @()
    foreach ($p in $paras) { $parts += (($p.runs | ForEach-Object { $_.t }) -join "") }
    $textRange2.Text = ($parts -join "`r")
    $textRange2.Font.Name = $defaults.font
    $textRange2.Font.Size = $defaults.size
    $textRange2.Font.Fill.ForeColor.RGB = (Rgb $defaults.color)
    $textRange2.ParagraphFormat.Bullet.Visible = 0
    if ($textRange2.Text.Length -eq 0) { return }
    $start = 1
    $i = 1
    foreach ($p in $paras) {
        $para = $textRange2.Paragraphs($i)
        $pf = $para.ParagraphFormat
        $pf.Alignment = @{ "l" = 1; "c" = 2; "r" = 3 }[[string]$p.align]
        $size = [double]$defaults.size
        foreach ($run in $p.runs) { if ($run.s -and [double]$run.s -gt $size) { $size = [double]$run.s } }
        $pf.LineRuleWithin = 0
        $pf.SpaceWithin = [double]$p.within * $size
        $pf.SpaceBefore = 0
        $pf.SpaceAfter = [double]$p.after
        $pf.LeftIndent = Pt $p.left
        $pf.FirstLineIndent = Pt $p.first
        foreach ($tab in $p.tabs) { [void]$pf.TabStops.Add(1, (Pt $tab)) }
        foreach ($run in $p.runs) {
            $n = $run.t.Length
            if ($n -gt 0) {
                $chars = $textRange2.Characters($start, $n)
                if ($run.f) { $chars.Font.Name = $run.f }
                if ($run.s) { $chars.Font.Size = [double]$run.s }
                if ($run.c) { $chars.Font.Fill.ForeColor.RGB = (Rgb $run.c) }
                $chars.Font.Bold = $(if ($run.b) { -1 } else { 0 })
                $chars.Font.Italic = $(if ($run.i) { -1 } else { 0 })
            }
            $start += $n
        }
        $start += 1
        $i += 1
    }
}

function Set-Frame($shape, $item) {
    $tf = $shape.TextFrame2
    $tf.AutoSize = 0
    $tf.WordWrap = -1
    $m = $item.margins
    $tf.MarginLeft = Pt $m[0]; $tf.MarginTop = Pt $m[1]; $tf.MarginRight = Pt $m[2]; $tf.MarginBottom = Pt $m[3]
    $tf.VerticalAnchor = @{ "top" = 1; "middle" = 3; "bottom" = 4 }[[string]$item.anchor]
}

function Set-Fill($shape, $item) {
    if ($item.fill) { $shape.Fill.Visible = -1; $shape.Fill.Solid(); $shape.Fill.ForeColor.RGB = (Rgb $item.fill) }
    else { $shape.Fill.Visible = 0 }
    if ($item.line) { $shape.Line.Visible = -1; $shape.Line.ForeColor.RGB = (Rgb $item.line); $shape.Line.Weight = Pt $item.line_mm }
    else { $shape.Line.Visible = 0 }
    $shape.Shadow.Visible = 0
}

$cfg = Get-Content -Raw -Encoding UTF8 $Spec | ConvertFrom-Json
$app = New-Object -ComObject PowerPoint.Application
$pres = $app.Presentations.Add(0)
try {
    $pres.PageSetup.SlideWidth = Pt $cfg.width_mm
    $pres.PageSetup.SlideHeight = Pt $cfg.height_mm
    $slide = $pres.Slides.Add(1, 12)
    $pres.SlideMaster.Theme.ThemeColorScheme.Colors(11).RGB = (Rgb "#ffffff")
    $pres.SlideMaster.Theme.ThemeColorScheme.Colors(12).RGB = (Rgb "#ffffff")
    $slide.FollowMasterBackground = 0
    $slide.Background.Fill.Solid()
    $slide.Background.Fill.ForeColor.RGB = (Rgb "#ffffff")
    $shapes = @{}
    $checks = @()

    foreach ($item in $cfg.items) {
        $top = $item.y
        if ($item.below) { $top = ($shapes[$item.below].Top + $shapes[$item.below].Height) / $PtPerMm + $item.gap }
        switch ($item.kind) {
            "rect" {
                $type = $(if ($item.shape -eq "round") { 5 } elseif ($item.shape -eq "oval") { 9 } else { 1 })
                $s = $slide.Shapes.AddShape($type, (Pt $item.x), (Pt $top), (Pt $item.w), (Pt $item.h))
                Set-Fill $s $item
                if ($type -eq 5) { $s.Adjustments.Item(1) = [double]$item.radius_mm / [Math]::Min($item.w, $item.h) }
                if ($item.paras) {
                    Set-Frame $s $item
                    Set-Paragraphs $s.TextFrame2.TextRange $item.paras $cfg.defaults
                }
            }
            "text" {
                $s = $slide.Shapes.AddTextbox(1, (Pt $item.x), (Pt $top), (Pt $item.w), (Pt $item.h))
                Set-Fill $s $item
                Set-Frame $s $item
                Set-Paragraphs $s.TextFrame2.TextRange $item.paras $cfg.defaults
                if ($item.link) {
                    $full = $s.TextFrame.TextRange.Text
                    $at = $full.IndexOf([string]$item.link.text)
                    if ($at -ge 0) {
                        $s.TextFrame.TextRange.Characters($at + 1, $item.link.text.Length).ActionSettings(1).Hyperlink.Address = $item.link.url
                    }
                }
            }
            "picture" {
                $s = $slide.Shapes.AddPicture([string]$item.path, 0, -1, (Pt $item.x), (Pt $top), (Pt $item.w), (Pt $item.h))
            }
            "line" {
                $s = $slide.Shapes.AddLine((Pt $item.x), (Pt $top), (Pt ($item.x + $item.w)), (Pt ($top + $item.h)))
                $s.Line.ForeColor.RGB = (Rgb $item.line); $s.Line.Weight = Pt $item.line_mm
            }
            "table" {
                $rows = $item.rows.Count; $cols = $item.cols.Count
                $s = $slide.Shapes.AddTable($rows, $cols, (Pt $item.x), (Pt $top), (Pt $item.w), (Pt 10))
                $tbl = $s.Table
                $tbl.ApplyStyle("{2D5ABB26-0587-4C30-8999-92F81FD0307C}", $false)
                for ($c = 1; $c -le $cols; $c++) { $tbl.Columns($c).Width = Pt $item.cols[$c - 1] }
                for ($r = 1; $r -le $rows; $r++) {
                    $row = $item.rows[$r - 1]
                    for ($c = 1; $c -le $cols; $c++) {
                        $cell = $tbl.Cell($r, $c)
                        $cs = $cell.Shape
                        $cs.Fill.Visible = 0
                        $tf = $cs.TextFrame2
                        $tf.MarginLeft = Pt $item.pad_x; $tf.MarginRight = Pt $item.pad_x
                        $tf.MarginTop = Pt $item.pad_y; $tf.MarginBottom = Pt $item.pad_y
                        Set-Paragraphs $tf.TextRange @($row.cells[$c - 1]) $cfg.defaults
                        foreach ($side in 1, 2, 4) { $cell.Borders($side).Visible = 0 }
                        $bottom = $cell.Borders(3)
                        $bottom.Visible = -1
                        $bottom.ForeColor.RGB = (Rgb $row.border)
                        $bottom.Weight = Pt $row.border_mm
                    }
                    $tbl.Rows($r).Height = Pt $item.row_min
                }
            }
        }
        $s.Name = [string]$item.name
        $shapes[[string]$item.name] = $s
    }

    # checks
    $min = 1000; $minWhere = ""
    $W = $pres.PageSetup.SlideWidth; $H = $pres.PageSetup.SlideHeight
    foreach ($s in $slide.Shapes) {
        if ($s.Left -lt -0.5 -or $s.Top -lt -0.5 -or ($s.Left + $s.Width) -gt ($W + 0.5) -or ($s.Top + $s.Height) -gt ($H + 0.5)) {
            $checks += "CHECK OFF-SLIDE $($s.Name)"
        }
        $frames = @()
        if ($s.HasTable) {
            foreach ($r in 1..$s.Table.Rows.Count) { foreach ($c in 1..$s.Table.Columns.Count) { $frames += $s.Table.Cell($r, $c).Shape.TextFrame2 } }
        } elseif ($s.HasTextFrame -and $s.TextFrame2.HasText) {
            $frames += $s.TextFrame2
            $tf = $s.TextFrame2
            $need = $tf.TextRange.BoundHeight + $tf.MarginTop + $tf.MarginBottom
            if ($need -gt $s.Height + 0.5) {
                $checks += ("CHECK OVERFLOW {0} by {1:N1} mm" -f $s.Name, (($need - $s.Height) / $PtPerMm))
            }
        }
        foreach ($tf in $frames) {
            foreach ($run in $tf.TextRange.Runs()) {
                if ($run.Text.Trim().Length -gt 0 -and $run.Font.Size -lt $min) { $min = $run.Font.Size; $minWhere = $s.Name }
            }
        }
    }
    $last = $null
    foreach ($s in $slide.Shapes) { if ($s.HasTable) { $last = $s } }
    if ($last) { "TABLE {0:N1} {1:N1}" -f ($last.Top / $PtPerMm), (($last.Top + $last.Height) / $PtPerMm) }
    foreach ($name in $cfg.report) {
        $s = $shapes[[string]$name]
        "BOX {0} top {1:N1} bottom {2:N1} mm" -f $name, ($s.Top / $PtPerMm), (($s.Top + $s.Height) / $PtPerMm)
    }
    $checks
    "MINFONT " + $min.ToString([Globalization.CultureInfo]::InvariantCulture) + " " + $minWhere

    $pres.SaveAs($Pptx, 24)
    $pres.SaveAs($Pdf, 32)
    "SAVED"
}
finally {
    $pres.Close()
    $app.Quit()
    [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($app)
}
