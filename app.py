import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 1. Configuración de la página web (Modo Ancho y Diseño Limpio)
st.set_page_config(layout="wide", page_title="SVD Image Compression", page_icon="🧮")

st.title("🧮 Compresión de Imágenes usando SVD")
st.markdown("### Una aplicación práctica de Álgebra Lineal Avanzada")
st.write("Esta aplicación demuestra el Teorema de Eckart-Young utilizando la Descomposición en Valores Singulares (SVD).")

# 2. Barra lateral para controles interactivos
st.sidebar.header("⚙️ Panel de Control")
archivo_subido = st.sidebar.file_uploader(
    "Sube una imagen (JPG, PNG)",
    type=["jpg", "png", "jpeg"]
)

if archivo_subido is not None:
    # Cargar la imagen utilizando PIL y transformarla a Escala de Grises (Matriz 2D)
    imagen_pil = Image.open(archivo_subido).convert('L')
    A = np.array(imagen_pil, dtype=float)

    # Obtener dimensiones de la matriz original
    alto, ancho = A.shape
    max_k = min(alto, ancho)

    # Control deslizante dinámico para seleccionar el rango K
    st.sidebar.markdown("---")
    k_seleccionado = st.sidebar.slider(
        "Número de Valores Singulares a mantener (k)",
        min_value=1,
        max_value=max_k,
        value=max(1, int(max_k * 0.05))
    )

    # Datos informativos para la defensa del proyecto
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Datos de la Matriz")
    st.sidebar.write(f"**Dimensión de A:** {alto} x {ancho}")
    st.sidebar.write(f"**Rango máximo:** {max_k}")

    # Cálculo del porcentaje de datos guardados
    datos_originales = alto * ancho
    datos_comprimidos = (
        (alto * k_seleccionado)
        + k_seleccionado
        + (ancho * k_seleccionado)
    )

    porcentaje_ahorro = (
        1 - (datos_comprimidos / datos_originales)
    ) * 100

    st.sidebar.write(f"**Espacio ahorrado:** {porcentaje_ahorro:.2f}%")

    # -----------------------------------------------------------------
    # NÚCLEO MATEMÁTICO: Descomposición en Valores Singulares (SVD)
    # -----------------------------------------------------------------

    U, S, VT = np.linalg.svd(A, full_matrices=False)

    # Truncamos las matrices para quedarnos solo con el rango 'k'
    U_k = U[:, :k_seleccionado]
    S_k = np.diag(S[:k_seleccionado])
    VT_k = VT[:k_seleccionado, :]

    # Reconstrucción de la matriz
    A_comprimida = np.dot(U_k, np.dot(S_k, VT_k))
    A_comprimida = np.clip(A_comprimida, 0, 255).astype(np.uint8)

    # -----------------------------------------------------------------
    # 3. Despliegue visual en dos columnas
    # -----------------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼️ Imagen Original (Matriz Completa A)")
        st.image(imagen_pil, use_container_width=True)

    with col2:
        st.subheader(
            f"📉 Imagen Comprimida (Rango Aproximado k = {k_seleccionado})"
        )
        st.image(A_comprimida, use_container_width=True)

    # -----------------------------------------------------------------
    # 4. Sustento Académico: Gráfico del Espectro
    # -----------------------------------------------------------------

    st.markdown("---")
    st.subheader(
        "📉 Sustento Matemático: Comportamiento de los Valores Singulares ($\\sigma$)"
    )

    st.write(
        "Los valores singulares están ordenados de mayor a menor. "
        "Observa cómo los primeros valores retienen casi toda la energía "
        "(información) de la imagen, cayendo drásticamente después."
    )

    fig, ax = plt.subplots(figsize=(10, 3.5))

    ax.plot(
        S,
        color="#00aaff",
        lw=2.5,
        label="Valores Singulares ($\\sigma_i$)"
    )

    ax.axvline(
        x=k_seleccionado,
        color="#ef4444",
        linestyle="--",
        lw=2,
        label=f"Corte actual (k = {k_seleccionado})"
    )

    ax.set_yscale('log')
    ax.set_xlabel("Índice del Valor Singular ($i$)")
    ax.set_ylabel("Magnitud (Escala Logarítmica)")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.4)

    st.pyplot(fig)

else:
    st.info(
        "👋 ¡Bienvenido! Para iniciar la demostración, "
        "arrastra o sube una imagen desde el panel izquierdo."
    )