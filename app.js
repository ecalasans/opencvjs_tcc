let fonte;
let histograma;

$(document).ready(function (){
    // Carrega imagem por vias normais

    $("#img_input").change(function (e) {
        fonte = document.getElementById("cont_imagem");
        fonte.src = URL.createObjectURL(e.target.files[0]);
    });
///////////////////////////////////////

// Manipulação com OpenCV
    $("#cont_imagem").load(function () {
        var mat = cv.imread(fonte);
        let w = mat.rows;
        let h = mat.cols;
        let vetor = new cv.MatVector();

        let dst = new cv.Mat(w, h, cv.CV_8UC1);

        cv.cvtColor(mat, dst, cv.COLOR_RGBA2GRAY, 0);  //Converte em escala de cinza

        //console.log(dst.type());
        //console.log(dst.ucharAt(120,120));  //Acessa o valor de um pixel

        //let retangulo = new cv.Rect(50,50, 100,120);  //Constroi um retangulo
       // let porcao = mat.roi(retangulo);  //Seleciona parte de uma imagem(parametro:  retanngulo)

        vetor.push_back(dst);

        //Cálculo do histograma
        let accumulate = false;
        let channels = [0];
        let hist_size = [256];
        let ranges = [0, 255];
        histograma = new cv.Mat();
        let mascara = new cv.Mat();

        cv.calcHist(vetor, channels, mascara, histograma, hist_size, ranges, accumulate);

        cv.imshow('img_canvas', dst);

        mat.delete();
       // retangulo.delete();
       // porcao.delete();

    });
////////////////////////////////////////
});

// Função de loading
function openCVReady(){
    $("#loader").remove();
}

