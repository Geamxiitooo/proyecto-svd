import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO

# --------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# --------------------------------

st.set_page_config(
    page_title="Compresión de Imágenes con SVD",
    page_icon="🧮",
    layout="wide"
)

st.title("🧮 Compresión de Imágenes mediante Descomposición en Valores Singulares")
st.markdown(
    """
    Esta aplicación demuestra cómo el Álgebra Lineal permite reducir
    la cantidad de información necesaria para representar una imagen
    utilizando aproximaciones de rango reducido mediante SVD.
    """
)
st.markdown(
    "### Aplicación de Álgebra Lineal mediante Descomposición en Valores Singulares (SVD)"
)

# --------------------------------
# FUNCIONES
# --------------------------------

def comprimir_canal(canal, k):
    U, S, VT = np.linalg.svd(canal, full_matrices=False)

    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    VT_k = VT[:k, :]

    comprimido = U_k @ S_k @ VT_k
    comprimido = np.clip(comprimido, 0, 255)

    return comprimido.astype(np.uint8), U, S, VT


def calcular_mse(original, comprimida):
    return np.mean(
        (original.astype(float) - comprimida.astype(float)) ** 2
    )


# --------------------------------
# BARRA LATERAL
# --------------------------------

st.sidebar.header("⚙️ Configuración")

archivo_subido = st.sidebar.file_uploader(
    "Sube una imagen",
    type=["jpg", "jpeg", "png"]
)

modo = st.sidebar.radio(
    "Modo de procesamiento",
    ["Escala de grises", "Color RGB"]
)

# --------------------------------
# PROCESAMIENTO
# --------------------------------

if archivo_subido:

    imagen_original = Image.open(archivo_subido)

    # --------------------------------
    # REDIMENSIONAMIENTO AUTOMÁTICO PARA MEJORAR EL RENDIMIENTO DEL PRGRAMA 
    # --------------------------------

    MAX_DIM = 800

    ancho_original, alto_original = imagen_original.size

    if max(ancho_original, alto_original) > MAX_DIM:

        imagen_original.thumbnail((MAX_DIM, MAX_DIM))

        st.warning(
            f"⚡ Imagen redimensionada automáticamente "
            f"para mejorar el rendimiento.\n\n"
            f"Nuevo tamaño: "
            f"{imagen_original.size[0]} x {imagen_original.size[1]}"
        )

    # ======================================
    # ESCALA DE GRISES
    # ======================================

    if modo == "Escala de grises":

        imagen_gris = imagen_original.convert("L")

        A = np.array(
            imagen_gris,
            dtype=float
        )

        alto, ancho = A.shape

        max_k = min(alto, ancho)

        k = st.sidebar.slider(
            "Valores singulares (k)",
            1,
            max_k,
            max(1, int(max_k * 0.05))
        )

        A_comprimida, U, S, VT = comprimir_canal(A, k)

        mse = calcular_mse(
            A,
            A_comprimida
        )

        datos_originales = alto * ancho

        datos_comprimidos = (
            (alto * k)
            + k
            + (ancho * k)
        )

        ahorro = (
            1
            - datos_comprimidos / datos_originales
        ) * 100

        st.sidebar.markdown("---")
        st.sidebar.write(f"📏 Dimensión: {alto} x {ancho}")
        st.sidebar.write(f"📉 Ahorro estimado: {ahorro:.2f}%")
        st.sidebar.write(f"📐 MSE: {mse:.2f}")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🖼️ Imagen Original")
            st.image(
                imagen_gris,
                use_container_width=True
            )

        with col2:
            st.subheader(
                f"📉 Imagen Comprimida (k={k})"
            )
            st.image(
                A_comprimida,
                use_container_width=True
            )

        imagen_descarga = Image.fromarray(
            A_comprimida
        )

    # ======================================
    # RGB
    # ======================================

    else:

        imagen_rgb = imagen_original.convert("RGB")

        img_array = np.array(imagen_rgb)

        R = img_array[:, :, 0]
        G = img_array[:, :, 1]
        B = img_array[:, :, 2]

        alto, ancho = R.shape

        max_k = min(alto, ancho)

        k = st.sidebar.slider(
            "Valores singulares (k)",
            1,
            max_k,
            max(1, int(max_k * 0.05))
        )

        R_c, U, S, VT = comprimir_canal(
            R,
            k
        )

        G_c, _, _, _ = comprimir_canal(
            G,
            k
        )

        B_c, _, _, _ = comprimir_canal(
            B,
            k
        )

        imagen_comprimida = np.stack(
            [R_c, G_c, B_c],
            axis=2
        )

        mse = calcular_mse(
            img_array,
            imagen_comprimida
        )

        datos_originales = alto * ancho * 3

        datos_comprimidos = 3 * (
            (alto * k)
            + k
            + (ancho * k)
        )

        ahorro = (
            1
            - datos_comprimidos / datos_originales
        ) * 100

        st.sidebar.markdown("---")
        st.sidebar.write(f"📏 Dimensión: {alto} x {ancho}")
        st.sidebar.write(f"📉 Ahorro estimado: {ahorro:.2f}%")
        st.sidebar.write(f"📐 MSE: {mse:.2f}")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🖼️ Imagen Original")
            st.image(
                imagen_rgb,
                use_container_width=True
            )

        with col2:
            st.subheader(
                f"📉 Imagen Comprimida (k={k})"
            )
            st.image(
                imagen_comprimida,
                use_container_width=True
            )

        imagen_descarga = Image.fromarray(
            imagen_comprimida
        )

    # --------------------------------
    # DESCARGA!!!
    # --------------------------------

    st.markdown("---")

    buffer = BytesIO()

    imagen_descarga.save(
        buffer,
        format="PNG"
    )

    st.download_button(
        label="⬇️ Descargar imagen comprimida",
        data=buffer.getvalue(),
        file_name="imagen_comprimida.png",
        mime="image/png"
    )

    # --------------------------------
    # GRÁFICO!!!
    # --------------------------------

    st.markdown("---")

    st.subheader(
        "📈 Comportamiento de los Valores Singulares"
    )

    fig, ax = plt.subplots(
        figsize=(10, 4)
    )

    ax.plot(
        S,
        linewidth=2
    )

    ax.axvline(
        x=k,
        linestyle="--",
        linewidth=2,
        label=f"k = {k}"
    )

    ax.set_yscale("log")
    ax.set_xlabel("Índice")
    ax.set_ylabel("Magnitud")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

    # --------------------------------
    # MATRICES SVD
    # --------------------------------

    st.markdown("---")

    st.subheader(
        "🧮 Matrices de la Descomposición SVD"
    )

    tab1, tab2, tab3 = st.tabs(
        ["U", "Σ", "Vᵀ"]
    )

    with tab1:

        st.write(
            "Primeras 5 filas y columnas de U"
        )

        st.dataframe(
            np.round(
                U[:5, :5],
                3
            )
        )

    with tab2:

        sigma = np.diag(
            S[:5]
        )

        st.write(
            "Primeros valores singulares"
        )

        st.dataframe(
            np.round(
                sigma,
                3
            )
        )

    with tab3:

        st.write(
            "Primeras 5 filas y columnas de Vᵀ"
        )

        st.dataframe(
            np.round(
                VT[:5, :5],
                3
            )
        )

else:

    st.info(
        "👋 Sube una imagen para comenzar con el programa."
    )
