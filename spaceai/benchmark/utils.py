from typing import List, Tuple

def merge_intervals(
    intervals: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """Merge overlapping intervals."""
    if not intervals:
        return []

    events = []
    for interval in intervals: 
        events.extend([(interval[0], 1), (interval[1], -1)]) # +1 for start, -1 for end of an interval
    
    events.sort(key=lambda e: (e[0], -e[1]))
    current_depth = 0
    curr_start = 0

    res = []
    for event in events:
        if current_depth == 0 and event[1] == 1:
            curr_start = event[0]

        current_depth += event[1]

        if current_depth == 0:
            res.append((curr_start, event[0]))

    return res