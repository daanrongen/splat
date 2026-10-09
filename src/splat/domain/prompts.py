from splat.domain.errors import SplatDomainError

Point = tuple[float, float, int]  # x, y, label (1 foreground, 0 background)
Box = tuple[float, float, float, float]  # x0, y0, x1, y1


def parse_points(texts: list[str]) -> list[Point]:
    """Parses 'x,y' or 'x,y,label' strings; the label defaults to foreground."""
    points = []
    for text in texts:
        try:
            x, y, *label = text.split(",")
            point = (float(x), float(y), int(label[0]) if label else 1)
        except (ValueError, IndexError):
            raise SplatDomainError(
                f"Invalid point {text!r}; expected 'x,y' or 'x,y,label'."
            ) from None
        if point[2] not in (0, 1) or len(label) > 1:
            raise SplatDomainError(f"Invalid point {text!r}; the label is 1 or 0.")
        points.append(point)
    return points


def parse_box(text: str) -> Box:
    try:
        x0, y0, x1, y1 = (float(part) for part in text.split(","))
    except ValueError:
        raise SplatDomainError(f"Invalid box {text!r}; expected 'x0,y0,x1,y1'.") from None
    if x1 <= x0 or y1 <= y0:
        raise SplatDomainError(f"Invalid box {text!r}; it needs x1 > x0 and y1 > y0.")
    return x0, y0, x1, y1
