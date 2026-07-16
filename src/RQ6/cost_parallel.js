const methods = ['Gen-Devign', 'Disc-Devign', 'Gen-Reveal', 'Disc-Reveal'];
const colors = ['#5470c6', '#fac858', '#91cc75', '#ee6666'];

const rawMetrics = [
  [19.81, 3.18, 3, 9.90, 18.34, 2.63],
  [15.42, 3.08, 3, 9.26, 15.98, 4.00],
  [19.22, 2.64, 4, 10.98, 18.34, 2.62],
  [12.73, 2.55, 3, 7.64, 15.98, 3.99]
];

const methodIndices = [0, 1, 2, 3];
methodIndices.sort((a, b) => rawMetrics[a][0] - rawMetrics[b][0]);

const sortedMethods = methodIndices.map(i => methods[i]);
const sortedColors = methodIndices.map(i => colors[i]);
const sortedMetrics = methodIndices.map(i => rawMetrics[i]);

const seriesData = sortedMethods.map((name, idx) => ({
  name: name,
  value: [idx, ...sortedMetrics[idx]],
  lineStyle: { color: sortedColors[idx], width: 8, opacity: 0.5 }
}));

option = {
  legend: {
    data: sortedMethods,
    bottom: 50,
    icon: 'circle',
    itemWidth: 10,
    itemHeight: 10
  },
  textStyle: { fontSize: 16 },
  parallelAxis: [
    {
      dim: 0,
      name: 'Method\nDataset',
      type: 'category',
      data: sortedMethods,
      axisTick: { show: false },
      axisLabel: { show: true },
      splitLine: { show: false },
      nameLocation: 'end',
      nameGap: 30
    },
    {
      dim: 1,
      name: 'Total Training\nTime (min)',
      min: 10,
      max: 22,
      nameLocation: 'end',
      nameGap: 40,
      axisLabel: { show: false },
      axisTick: { show: false },
      splitLine: { show: false }
    },
    {
      dim: 2,
      name: 'Train Time/\nEpoch (min)',
      min: 2.2,
      max: 3.3,
      nameLocation: 'end',
      nameGap: 40,
      axisLabel: { show: false },
      axisTick: { show: false },
      splitLine: { show: false }
    },
    {
      dim: 3,
      name: 'Convergence\nEpoch',
      min: 1,
      max: 5,
      nameLocation: 'end',
      nameGap: 40,
      axisLabel: { show: false },
      axisTick: { show: false },
      splitLine: { show: false }
    },
    {
      dim: 4,
      name: 'Converge\nTime (min)',
      min: 6,
      max: 12,
      nameLocation: 'end',
      nameGap: 40,
      axisLabel: { show: false },
      axisTick: { show: false },
      splitLine: { show: false }
    },
    {
      dim: 5,
      name: 'Peak Training\nMemory (GB)',
      min: 0,
      max: 24,
      nameLocation: 'end',
      nameGap: 40,
      axisLabel: { show: false },
      axisTick: { show: false },
      splitLine: { show: false }
    },
    {
      dim: 6,
      name: 'Inference\nLatency (ms)',
      min: 2,
      max: 4,
      nameLocation: 'end',
      nameGap: 40,
      axisLabel: { show: false },
      axisTick: { show: false },
      splitLine: { show: false }
    }
  ],
  parallel: {
    left: '10%',
    right: '12%',
    top: '15%',
    bottom: '12%',
    parallelAxisDefault: {
      axisLine: { lineStyle: { color: '#aaa' } },
      nameTextStyle: { fontSize: 14, fontWeight: 'bold', color: '#555' }
    }
  },
  series: {
    type: 'parallel',
    lineStyle: { width: 8, opacity: 0.5 },
    data: seriesData,
    label: {
      show: true,
      formatter: (params) => {
        if (params.dimensionIndex === 0) {
          return params.seriesName;
        }
        return params.value[params.dimensionIndex];
      },
      position: 'top',
      offset: [0, -8],
      fontSize: 11,
      color: '#333'
    }
  }
};