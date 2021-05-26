const cv = require('./js/opencv');
require('jquery');
const funcoes = require('./js/constants');
const $ = require('jquery-browserify');

$(document).ready(function (){
    let histograma;
    let mat;
    let dst;
    // Carrega imagem por vias normais

    $("#img_input").change(function (e) {
        fonte = document.getElementById("cont_imagem");
        fonte.src = URL.createObjectURL(e.target.files[0]);
    });
///////////////////////////////////////

// Manipulação com OpenCV
    $("#cont_imagem").load(function () {
        mat = cv.imread(fonte);
        let w = mat.rows;
        let h = mat.cols;
        let vetor = new cv.MatVector();

        dst = new cv.Mat(w, h, cv.CV_8UC1);


        cv.cvtColor(mat, dst, cv.COLOR_RGBA2GRAY, 0);  //Converte em escala de cinza
        //console.log(dst.data[120,120]);
        //console.log(dst.ucharAt(120,120));  //Acessa o valor de um pixel

        let retangulo = new cv.Rect(50,50, 100,50);  //Constroi um retangulo
        let porcao = dst.roi(retangulo);  //Seleciona parte de uma imagem(parametro:  retanngulo)
        console.log(porcao.rows, porcao.cols);

        vetor.push_back(dst);

        //Cálculo do histograma
        let accumulate = false;
        let channels = [0];
        let hist_size = [256];
        let ranges = [0, 255];
        histograma = new cv.Mat();
        let mascara = new cv.Mat();

        cv.calcHist(vetor, channels, mascara, histograma, hist_size, ranges, accumulate);

        cv.imshow('img_canvas', porcao);

        //mat.delete();
        // retangulo.delete();
        // porcao.delete();

    });

    $("#calc_fft").click(function (e){
        let im = funcoes.GaussModif(
            0.45,
            2.0,
            4.0,
            2503, dst);


    });

////////////////////////////////////////
});




