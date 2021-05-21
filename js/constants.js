const nd = require('nd4js');
const cv = require('./opencv');

// Função para compor a meshgrid
function MeshgridJS(xdim, ydim){
    let x = new Array(xdim);
    let y = new Array(xdim);
    let x_dim = xdim;
    let y_dim = ydim;

    for (let i = 0; i < xdim; i++){
        let temp1 = new Array(ydim);

        for (let j = 0; j < ydim; j++){
            temp1[j] = i;
        }

        x[i] = temp1;
    }

    for (let i = 0; i < xdim; i++){
        let temp2 = new Array(ydim);

        for (let j = 0; j < ydim; j++){
            temp2[j] = j;
        }

        y[i] = temp2;
    }
    return {
        'x': x,
        'y': y,
        'u': x_dim,
        'v': y_dim
    };
}

//Função Gaussiana Modificaca
function GaussModif(gamma_l = 0.0, gamma_h = 0.0, c = 0.0, D0 = 0.0, imagem) {
    //Dimensões da imagem
    let im_h = imagem.rows;
    let im_w = imagem.cols;

    //Coordenadas do centro
    let u_c = im_h/2;
    let v_c = im_w/2;

    //Matriz de Coordenadas
    let arr = MeshgridJS(im_h,im_w);
    let a = arr['x'];
    let b = arr['y'];

    //Etapas de cálculo de H(u,v)
    let d_uv_2_d0 = nd.zip_elems([a,b], (u, v, i, j) => (((u-u_c)**2 + (v-v_c)**2)**2) / D0);
    let um_menos = nd.zip_elems([d_uv_2_d0], (hij,i,j) => (Math.exp((-1) * c * hij)));
    let expon = nd.zip_elems([um_menos], (mij, i, j) => 1 - mij);
    let multi_delta_gamma = nd.zip_elems([expon], (mij, i, j) => (gamma_h - gamma_l) * mij);

    let H_uv = nd.zip_elems([multi_delta_gamma], (mij, i, j) => gamma_l + mij)

    console.log(H_uv);
}

// Função para zero padding
function ZeroPadding(imagem) {
    //Calcula o tamanho ótimo para a FFT
    let im_h = cv.getOptimalDFTSize(imagem.rows);
    let im_w = cv.getOptimalDFTSize(imagem.cols);

    // Matriz de destino para a imagem modificada
    let z_padded = new cv.Mat();

    // Valor escalar representando pixels pretos
    let preto = new cv.Scalar.all(0);

    // Constrói a imagem modificada
    cv.copyMakeBorder(
        imagem, z_padded,
        0, im_h - imagem.rows, 0, im_w - imagem.cols,
        cv.BORDER_CONSTANT, preto
    );

    return z_padded;
}

// Função que retira os pixels extras adicionados por ZeroPadding
function ZeroUnpadding(imagem, padded_imagem) {
    // Dimensões da imagem original
    let im_h = imagem.rows;
    let im_w = imagem.cols;

    let mascara = new cv.Rect(0,0, im_w, im_h);

    return padded_imagem.roi(mascara);

}

//Função para adaptar a matriz da imagem com um formato de matriz complexa(parte real e imaginária com valor 0i)
function PrepareToDFT(padded_imagem){
    let vetor = new cv.MatVector();
    let parte_real = new cv.Mat();
    padded_imagem.convertTo(parte_real, cv.CV_32F);
    let parte_imaginaria = new cv.Mat.zeros(padded_imagem.rows, padded_imagem.cols, cv.CV_32F);
    let complexa = new cv.Mat();
    vetor.push_back(parte_real);
    vetor.push_back(parte_imaginaria);
    cv.merge(vetor, complexa);

    return complexa;
}

//Função que troca os quadrantes das diagonais principal e secundária da imagem
function CrossQuads(imagem){
    let u_c = imagem.rows/2;
    let v_c = imagem.cols/2;

    let r1 = new cv.Rect(0,0, v_c,u_c);
    let r2 = new cv.Rect(v_c,0, v_c,u_c);
    let r3 = new cv.Rect(0,u_c, v_c,u_c);
    let r4 = new cv.Rect(v_c,u_c, v_c,u_c);

   //Pedaços da imagem
    let q1 = imagem.roi(r1);
    let q2 = imagem.roi(r2);
    let q3 = imagem.roi(r3);
    let q4 = imagem.roi(r4);

    //Troca os quadrntes
    let container = new cv.Mat();

    // 1 <-> 4
    q1.copyTo(container);
    q4.copyTo(q1);
    container.copyTo(q4);

    // 2 <-> 3
    q2.copyTo(container);
    q3.copyTo(q2);
    container.copyTo(q3);
}

//Função que calcula a FFT da imagem e retorna a matriz da imagem já com os quadrantes trocados
// pronta para a plotagem.
function MakeFFT(imagem) {
    // Otimiza a imagem para o cálculo da FFT
    let im_otim = ZeroPadding(imagem);

    //Transforma a imagem numa matriz complexa
    let im_compl = PrepareToDFT(im_otim);

    //Calcula a FFT
    let im_fft = new cv.Mat();
    cv.dft(im_compl, im_fft, cv.DFT_COMPLEX_OUTPUT);

    //Separa a parte real e imaginária da matriz complexa
    let componentes = new cv.MatVector();
    cv.split(im_fft, componentes);
    let re = componentes.get(0);
    let im = componentes.get(1);

    // Calcula o espectro
    let espectro = new cv.Mat();
    cv.magnitude(re, im, espectro);

    //Calcula log(1 + magnitude)
    let m1 = new cv.Mat.ones(espectro.rows, espectro.cols, espectro.type());
    let mag = espectro;
    cv.add(mag, m1, mag);
    cv.log(mag, mag);
    mag.convertTo(mag, cv.CV_8U);
    cv.normalize(mag, mag, 0, 255, cv.NORM_MINMAX);

    return mag;
}

module.exports = {
    MeshgridJS,
    GaussModif,
    ZeroPadding,
    ZeroUnpadding,
    PrepareToDFT,
    CrossQuads,
    MakeFFT
}