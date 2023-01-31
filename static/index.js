var chart = LightweightCharts.createChart(document.getElementById('chart'), {
	width: 1000,
  	height: 500,
	crosshair: {
		mode: LightweightCharts.CrosshairMode.Normal,
	},
});

var candleSeries = chart.addCandlestickSeries({
	upColor: '#00ff00',
	downColor: '#ff0000', 
	borderDownColor: '#ff0000',
	borderUpColor: '#00ff00',
	wickDownColor: '#ff0000',
	wickUpColor: '#00ff00',
});


fetch('http://localhost:5000/history')
	.then((r) => r.json())
	.then((response) => {
		console.log(response)

		candleSeries.setData(response);
	})

fetch('http://127.0.0.1:5000/history')
	.then((r) => r.json())
	.then((response) => {
		console.log(response)

		candleSeries.setData(response);
	})

var binanceSocket = new WebSocket("wss://stream.binance.com:9443/ws/gbpusdt@kline_15m");

binanceSocket.onmessage = function (event) {	
	var message = JSON.parse(event.data);

	var candlestick = message.k;

	console.log(candlestick)

	candleSeries.update({
		time: candlestick.t / 1000,
		open: candlestick.o,
		high: candlestick.h,
		low: candlestick.l,
		close: candlestick.c
	})
}