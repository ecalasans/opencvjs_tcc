const nd = require('nd4js');
const cv = require('./opencv');

// Função para compor a meshgrid
function MeshgridJS(xdim, ydim){
    let u = new Array();
    let v = new Array();
    //let x_dim = xdim;
    //let y_dim = ydim;

    for (let i = 0; i < xdim; i++){
        for (let j = 0; j < ydim; j++){
            u.push(i);
        }
    }

    for (let i = 0; i < xdim; i++){
        for (let j = 0; j < ydim; j++){
            v.push(j);
        }
    }
    return {
        'u': u,
        'v': v
    };
}

//Função Gaussiana Modificaca
function GaussModif(gamma_l = 0.0, gamma_h = 0.0, c = 0.0, D0 = 0.0, imagem) {
    //Calcula o tamanho ótimo para a FFT
    let im_h = cv.getOptimalDFTSize(imagem.rows);
    let im_w = cv.getOptimalDFTSize(imagem.cols);

    //Coordenadas do centro
    let u_c = new cv.Mat(im_h, im_w, cv.CV_32F, new cv.Scalar(im_h/2));
    let v_c = new cv.Mat(im_h, im_w, cv.CV_32F, new cv.Scalar(im_w/2));

    // Inclinacao da curva
    let m_c = new cv.Mat(im_h, im_w, cv.CV_32F, new cv.Scalar(c));

    // Frequencia de corte
    let m_d0 = new cv.Mat(im_h, im_w, cv.CV_32F, new cv.Scalar(D0));

    //Matriz de Coordenadas
    let arr = MeshgridJS(im_h,im_w);
    let a = arr['u'];
    a = cv.matFromArray(im_h, im_w, cv.CV_32F, a);
    let b = arr['v'];
    b = cv.matFromArray(im_h, im_w, cv.CV_32F, b);

    // Matriz do filtro gaussiano modificado
    let H_uv = new cv.Mat.zeros(im_h, im_w, cv.CV_32F);


    //Etapas de cálculo de H(u,v)
    // let d_uv_2_d0 = nd.zip_elems([a, b], (u, v, i, j) => (((u-u_c)**2 + (v-v_c)**2)**2) / D0);
    // 1.  Cálculo de D(u,v)/D0
    let u_uc = new cv.Mat();
    let v_vc = new cv.Mat();
    // (u - u_c)^2
    cv.subtract(a, u_c, u_uc);
    cv.multiply(u_uc, u_uc, u_uc)
    a.delete();

    // (v - v_c)^2
    cv.subtract(b, v_c, v_vc);
    cv.multiply(v_vc, v_vc, v_vc)
    b.delete();

    // [(u_uc)^2 + (v_vc)^2]^2
    let soma_uuc_vvc_2 = new cv.Mat();
    cv.add(u_uc, v_vc, soma_uuc_vvc_2);
    cv.multiply(soma_uuc_vvc_2, soma_uuc_vvc_2, soma_uuc_vvc_2);
    u_uc.delete();
    v_vc.delete();

    // Divisão por D0^2
    let d_d0 = new cv.Mat();
    cv.multiply(m_d0, m_d0, m_d0);
    cv.divide(soma_uuc_vvc_2, m_d0, d_d0);
    soma_uuc_vvc_2.delete();
    m_d0.delete();

    // 2.  Cálculo de -c * d_d0
    // let um_menos = nd.zip_elems([d_uv_2_d0], (hij,i,j) => (Math.exp((-1) * c * hij)));
    let menos_um = new cv.Mat(im_h, im_w, cv.CV_32F, new cv.Scalar(-1));
    let menos_c = new cv.Mat();

    // -c
    cv.multiply(menos_um, m_c, menos_c);

    cv.multiply(d_d0, menos_c, menos_c);
    menos_um.delete();
    d_d0.delete();

    // 3.  Cálculo de 1 - exp(menos_c)
    // let expon = nd.zip_elems([um_menos], (mij, i, j) => 1 - mij);

    // Matriz de "uns"
    let um = new cv.Mat.ones(im_h, im_w, cv.CV_32F);

    // Exponencial
    let exponencial = new cv.Mat();
    cv.exp(menos_c, exponencial);
    menos_c.delete()

    // 1 - exponencial
    let um_menos_exp = new cv.Mat();
    cv.subtract(um, exponencial, um_menos_exp);
    exponencial.delete();

    // 4.  Cálculo de H(u,v)
    // let multi_delta_gamma = nd.zip_elems([expon], (mij, i, j) => (gamma_h - gamma_l) * mij);
    // let H_uv = nd.zip_elems([multi_delta_gamma], (mij, i, j) => gamma_l + mij)
    let m_gamma_l = new cv.Mat(im_h, im_w, cv.CV_32F, new cv.Scalar(gamma_l));
    let m_gamma_h = new cv.Mat(im_h, im_w, cv.CV_32F, new cv.Scalar(gamma_h));
    let gh_gl = new cv.Mat();

    // gamma_h - gamma_l
    cv.subtract(m_gamma_h, m_gamma_l, gh_gl);

    // m_uv = (um_menos_exp * gh_gl) + gamma_l
    let m_huv = new cv.Mat();
    cv.multiply(gh_gl, um_menos_exp, gh_gl);
    cv.add(gh_gl, m_gamma_l, m_huv);
    m_gamma_l.delete();
    m_gamma_h.delete();
    gh_gl.delete();
    um_menos_exp.delete();

    console.log(m_huv);

    return m_huv;
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

    // Troca os quadrantes para visualização da DFT
    CrossQuads(mag);

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