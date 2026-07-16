const seriesData = [
  {
    value: [19.8, 9.9, 18.3, 4.5, 2.6],
    name: 'Gen-Devign',
    lineStyle: { color: '#5470c6', width: 2 },
    itemStyle: { color: '#5470c6' },
    areaStyle: { color: 'rgba(84,112,198,0.2)' }
  },
  {
    value: [19.2, 11.0, 18.3, 4.5, 2.6],
    name: 'Gen-Reveal',
    lineStyle: { color: '#91cc75', width: 2 },
    itemStyle: { color: '#91cc75' },
    areaStyle: { color: 'rgba(145,204,117,0.2)' }
  },
  {
    value: [15.4, 9.3, 16.0, 2.7, 4.0],
    name: 'Disc-Devign',
    lineStyle: { color: '#fac858', width: 2 },
    itemStyle: { color: '#fac858' },
    areaStyle: { color: 'rgba(250,200,88,0.2)' }
  },
  {
    value: [12.7, 7.6, 16.0, 2.7, 4.0],
    name: 'Disc-Reveal',
    lineStyle: { color: '#ee6666', width: 2 },
    itemStyle: { color: '#ee6666' },
    areaStyle: { color: 'rgba(238,102,100,0.2)' }
  }
];

const offsets = [
  [0, 0],   // Gen-Devign
  [0, 0],    // Gen-Reveal
  [0, 0],   // Disc-Devign
  [0, 0]     // Disc-Reveal
];

seriesData.forEach((item, seriesIndex) => {
  item.label = {
    show: true,
    formatter: (params) => params.value.toFixed(1),
    position: 'top',
    color: '#333',
    fontSize: 15,
    fontWeight: 500,
    offset: offsets[seriesIndex]
  };
});

option = {
  backgroundColor: '#fff',
  legend: {
    data: ['Gen-Devign', 'Gen-Reveal', 'Disc-Devign', 'Disc-Reveal'],
    bottom: 60,
    itemGap: 20,
    itemWidth: 18,
    itemHeight: 10,
    textStyle: { fontSize: 18 }
  },
  radar: {
    indicator: [
      { name: 'Total Training Time (min)', min: 5, max: 20 }, 
      { name: 'Converge Time (min)', min: 3, max: 11 },        
      { name: 'Peak Training Memory (GB)', min: 10, max: 20 }, 
      { name: 'Peak Inference Memory (GB)', min: 0, max: 6 }, 
      { name: 'Inference Latency (ms)', min: 0, max: 4}
    ],
    radius: '62%',
    axisName: {
      color: '#333',
      fontSize: 17,
      fontWeight: 'bold',
      padding: [0, 10, 25, 10]
    },
    nameGap: 8,
    splitNumber: 5,
    splitLine: { lineStyle: { color: '#e0e0e0' } },
    axisLine: { lineStyle: { color: '#ccc' } }
  },
  series: [
    {
      type: 'radar',
      data: seriesData,
      symbol: 'circle',
      symbolSize: 6
    }
  ]
};