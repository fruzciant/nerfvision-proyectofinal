### **Instrucciones para la Elaboración del Proyecto Final**
**Asignatura: Visión por Computador**

#### **Objetivo del proyecto:**
El propósito del proyecto final es aplicar de forma integrada los conocimientos de la asignatura mediante el desarrollo de una aplicación de visión por computador con un **pipeline claramente estructurado** de procesamiento y reconocimiento de imágenes o video. El tema es de libre elección, pero debe evidenciar dominio de las técnicas vistas en clase, diseño modular del sistema y una implementación funcional sobre datos reales.

---

### **1. Entrega de propuesta (obligatoria y previa al desarrollo):**

**Contenido mínimo:**

* **Título del proyecto**
* **Descripción del problema o aplicación a desarrollar** (qué se quiere detectar, reconocer o analizar, y en qué contexto)
* **Objetivos del proyecto** (general y específicos)
* **Descripción del pipeline de visión por computador**, indicando cada etapa usada con su entrada, salida y técnica o algoritmo a utilizar:
  1. **Adquisición** – fuente de datos (imágenes estáticas, video en tiempo real, dataset, cámara, etc.)
  2. **Preprocesamiento** – operaciones de normalización, redimensionamiento, conversión de espacio de color, filtrado, etc.
  3. **Extracción de características o segmentación** – detección de bordes, keypoints, regiones de interés, descriptores, etc.
  4. **Reconocimiento o clasificación** – modelo o algoritmo empleado (clasificador clásico, red neuronal, detector de objetos, etc.)
  5. **Postprocesamiento y visualización** – anotación de resultados sobre la imagen/video, métricas de evaluación, salida al usuario
* **Dataset o fuente de imágenes/video** que se utilizará, con justificación de su pertinencia
* **Descripción de la interfaz de usuario** (framework elegido y flujo básico de interacción con el pipeline)
* **Dependencias y entorno de ejecución** esperados (bibliotecas, versiones, GPU/CPU)

La propuesta debe ser aprobada por el docente antes de iniciar el desarrollo. Se valorará la pertinencia del problema, la viabilidad del pipeline propuesto y la claridad técnica de cada etapa.

---

### **2. Requisitos del proyecto final:**

#### 2.1 Pipeline de visión por computador
* El sistema debe implementar las **cinco etapas del pipeline** descritas en la propuesta: adquisición, preprocesamiento, extracción de características/segmentación, reconocimiento/clasificación, y postprocesamiento/visualización.
* Cada etapa debe estar implementada en un módulo independiente con responsabilidad única (cohesión alta, acoplamiento bajo).
* El pipeline debe soportar al menos uno de los siguientes modos de entrada: imágenes individuales, lotes de imágenes, o video (archivo o cámara en tiempo real).
* Se debe documentar en el informe el efecto de cada etapa sobre los datos, acompañado de ejemplos visuales.

#### 2.2 Reconocimiento o clasificación
* La aplicación debe incluir al menos un componente de **reconocimiento, detección o clasificación** sobre las imágenes o frames procesados (p. ej., clasificación de objetos, detección de rostros, reconocimiento de gestos, segmentación semántica, estimación de pose, etc.).
* Se debe justificar la elección del modelo o algoritmo (precisión esperada, velocidad, disponibilidad de datos de entrenamiento, etc.).
* Si se emplea un modelo preentrenado, se debe documentar su origen, arquitectura y condiciones de uso.

#### 2.3 Interfaz de usuario
* La aplicación debe contar con una interfaz que permita al usuario cargar imágenes o video, ejecutar el pipeline y visualizar los resultados de cada etapa.
* El framework es de **libre elección** (tkinter, PyQt, PySide, Streamlit, Gradio, web, etc.).
* La interfaz debe mostrar, al menos, la imagen de entrada y el resultado anotado con las detecciones o clasificaciones obtenidas.

#### 2.4 Evaluación del sistema
* El proyecto debe incluir una evaluación cuantitativa del desempeño del pipeline sobre un conjunto de imágenes o video de prueba.
* Las métricas deben ser apropiadas al tipo de tarea (exactitud, precisión, recall, F1, IoU, FPS, etc.) y reportarse en el informe.

---

### **3. Entregables:**

El único entregable es el **repositorio en GitHub** (asignado a través de GitHub Classroom), debidamente organizado e incluyendo:

1. `README.md` claro y completo (descripción del problema, instrucciones de instalación y ejecución, descripción del pipeline)
2. **Código fuente completo** organizado por etapas del pipeline, con separación clara de responsabilidades
3. **Dataset o instrucciones para obtenerlo** (si es público, enlace y script de descarga; si es propio, incluirlo en `data/`)
4. **Diagrama del pipeline** (imagen o formato editable) en la carpeta `docs/`
5. Documento resumen del proyecto (`.pdf`) con:
   - Introducción, motivación y objetivos
   - Descripción detallada de cada etapa del pipeline con ejemplos visuales
   - Justificación del modelo o algoritmo de reconocimiento/clasificación
   - Resultados, métricas de evaluación y análisis de errores
   - Conclusiones y trabajo futuro
6. `requirements.txt` con las dependencias Python (incluyendo versiones)
7. `.gitignore` apropiado

**Importante:** No se aceptarán entregas ni modificaciones después de la fecha de cierre publicada en GitHub Classroom. La hora de corte será estricta y estará especificada en horario UTC.

---

### **4. Criterios de evaluación:**

| Criterio | Descripción |
|----------|-------------|
| Pipeline de visión | Implementación completa y correcta de las cinco etapas; modularidad y coherencia entre ellas |
| Reconocimiento/clasificación | Elección justificada del modelo o algoritmo; correcta integración en el pipeline |
| Evaluación cuantitativa | Métricas apropiadas, análisis de resultados y discusión de errores |
| Interfaz de usuario | Funcionalidad, usabilidad y capacidad de visualizar resultados del pipeline |
| Documentación | Calidad del README, del PDF y del diagrama del pipeline |
| Calidad del código | Legibilidad, organización modular y reproducibilidad del entorno |
