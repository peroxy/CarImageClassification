async function predict() {
    const localPath = document.getElementById("path").value;
    const response = await fetch('http://127.0.0.1:5000/predict/' + localPath);
    //extract JSON from the http response
    const json = await response.json();


    draw_chart('Starost avtomobila', json.age, "agePlot", false);
    draw_chart('Barva avtomobila', json.color, "colorPlot", true);
    draw_chart('Število vrat', json.doors, "doorsPlot", false);
    draw_chart('Vrsta goriva', json.engine, "enginePlot", true);
    draw_chart('Oblika karoserije', json.body_shape, "bodyShapePlot", true);

}

function toPercentage(number) {
    return (number * 100).toFixed(2);
}

function draw_chart(title, model, id, rotate) {
    var myChart = echarts.init(document.getElementById(id));

    var age_x_axis = model.default.map(x => x[0]);

    var option = {
        title: {
            left:50,
            text: title
        },
        grid: {
            left: 75,
            top: 50,
            right: 0,
            bottom: 75
        },
        tooltip: {
            formatter: function (data) {
                return data.marker + data.seriesName + " model : " + data.value + "% probability";
            }
        },
        legend: {},
        xAxis: {
            data: model.default.map(x => x[0]),
            type: "category",
            axisLabel: {
                interval: 0,
                rotate: rotate ? 20 : 0 //If the label names are too long you can manage this by rotating the label.
            }
        },
        yAxis: {
            min: 0,
            max: 100
        },
        series: [
            {
                name: 'Default',
                type: 'bar',
                data: model.default.map(x => toPercentage(x[1])),
            },
            {
                name: 'Optimized',
                type: 'bar',
                data: model.optimized.map(x => toPercentage(x[1])),
            },
            {
                name: 'Pre-trained',
                type: 'bar',
                data: model.pretrained.map(x => toPercentage(x[1])),
            }
        ]
    };

    // use configuration item and data specified to show chart
    myChart.setOption(option);
}
