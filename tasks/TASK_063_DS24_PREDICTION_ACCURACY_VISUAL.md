# TASK_063: DS24 Prediction Accuracy Visual

## Status: COMPLETE

## Priority: Low (UI polish)

## Objective

Enhance "MCTS Root Value vs Actual Reward" chart with accuracy reference lines.

## Current State

Chart shows scatter plot of predicted value (x) vs actual reward (y).
Hard to visually assess prediction quality without reference.

## Requested Features

### 1. Perfect Prediction Line (100% accuracy)

Add diagonal line where `y = x` (predicted = actual).

```javascript
// Add to chart config
{
    type: 'line',
    data: {
        datasets: [{
            label: 'Perfect Prediction',
            data: [{x: min, y: min}, {x: max, y: max}],
            borderColor: 'rgba(0, 255, 0, 0.8)',
            borderDash: [5, 5],
            pointRadius: 0,
            fill: false
        }]
    }
}
```

### 2. 80% Accuracy Zone (shaded region)

Add shaded area showing ±20% tolerance band around perfect line.

```javascript
// Upper bound: y = 1.2x
// Lower bound: y = 0.8x
{
    label: '80% Accuracy Zone',
    backgroundColor: 'rgba(0, 255, 0, 0.1)',
    fill: 'between'  // or use two datasets with fill between
}
```

## Visual Mockup

```
    Actual
    Reward
       ^
       |      /  <- 100% line (green dashed)
       |    /////
       |  /////  <- 80% zone (light green fill)
       |/////
       +---------> Predicted Value
```

## Implementation Notes

- Use Chart.js annotation plugin or manual dataset for lines
- 80% zone could be two line datasets with `fill: '+1'` or `fill: '-1'`
- Calculate min/max from data range for line endpoints
- Keep existing scatter points on top of reference visuals

## Files to Modify

| File | Change |
|------|--------|
| `tools/DS24/dashboard.js` | Add reference line and zone to chart |

## Acceptance Criteria

- [x] Green dashed diagonal line showing y=x (perfect prediction)
- [x] Light green shaded zone showing ±20% tolerance
- [x] Scatter points remain visible on top (order=1)
- [x] Reference lines added dynamically based on data range
