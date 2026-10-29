/**
 * Cliente de la API de clasificación de residuos.
 *
 * El backend corre en la laptop; el teléfono lo alcanza por la IP de la red
 * local. `localhost` no sirve desde el móvil porque apunta al propio móvil.
 */

export type ColorKey =
  | 'azul'
  | 'amarillo'
  | 'blanco'
  | 'gris'
  | 'marron'
  | 'rojo'
  | 'negro';

export type Contenedor = {
  color: ColorKey;
  nombre: string;
  residuos: string;
  hex: string;
  texto_hex: string;
};

export type Candidato = {
  material: string;
  label: string;
  color: ColorKey;
  probabilidad: number;
};

export type Prediccion = {
  material: string;
  material_label: string;
  confianza: number;
  incierto: boolean;
  contenedor: Contenedor;
  consejo: string;
  top: Candidato[];
  ms: number;
  mensaje: string | null;
};

export type Salud = {
  estado: 'ok' | 'sin_modelo';
  detalle: string | null;
  modelo: { arch: string; clases: string[]; device: string } | null;
  max_mb: number;
};

/**
 * Error con mensaje listo para mostrar.
 *
 * `clase` separa "no hubo conexión" de "el servidor contestó un error": sin esa
 * distinción, un fallo de red y un rechazo del servidor se ven igual en
 * pantalla y no hay por dónde empezar a mirar.
 */
export type ClaseError = 'red' | 'timeout' | 'servidor';

export class ApiError extends Error {
  constructor(
    message: string,
    readonly clase: ClaseError,
    readonly causa?: string,
  ) {
    super(message);
  }
}

const TIMEOUT_MS = 20_000;

export function normalizar(baseUrl: string): string {
  const url = baseUrl.trim().replace(/\/+$/, '');
  return /^https?:\/\//.test(url) ? url : `http://${url}`;
}

async function pedir<T>(baseUrl: string, ruta: string, init?: RequestInit): Promise<T> {
  // AbortController evita que la app se quede colgada si la laptop está
  // apagada o la IP cambió: sin esto, fetch espera indefinidamente.
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
  try {
    const res = await fetch(`${normalizar(baseUrl)}${ruta}`, {
      ...init,
      signal: ctrl.signal,
    });
    if (!res.ok) {
      let detalle = `HTTP ${res.status}`;
      try {
        const cuerpo = await res.json();
        if (cuerpo?.detail) detalle = String(cuerpo.detail);
      } catch {
        // respuesta sin JSON: se queda el código de estado
      }
      throw new ApiError(detalle, 'servidor', `HTTP ${res.status}`);
    }
    return (await res.json()) as T;
  } catch (err) {
    if (err instanceof ApiError) throw err;
    if (err instanceof Error && err.name === 'AbortError') {
      throw new ApiError(
        'El servidor no respondió a tiempo.',
        'timeout',
        `sin respuesta en ${TIMEOUT_MS / 1000} s`,
      );
    }
    throw new ApiError(
      'No se pudo conectar con el servidor.',
      'red',
      err instanceof Error ? err.message : String(err),
    );
  } finally {
    clearTimeout(timer);
  }
}

export function obtenerSalud(baseUrl: string): Promise<Salud> {
  return pedir<Salud>(baseUrl, '/api/health');
}

/** ¿Está vivo el backend? Se usa para saber si un fallo fue de red o de la subida. */
export async function alcanzable(baseUrl: string): Promise<boolean> {
  try {
    await obtenerSalud(baseUrl);
    return true;
  } catch {
    return false;
  }
}

/**
 * Envía la foto ya codificada en base64.
 *
 * No se usa `FormData`: desde React Native 0.88 su stack de red rechaza el
 * objeto `{uri, name, type}` con el que se armaba el multipart y falla con
 * "Unsupported FormDataPart implementation"; exige un Blob real, que no hay
 * forma directa de obtener a partir de un `file://` del teléfono. Un JSON con
 * la imagen en base64 evita el problema entero. El backend sigue aceptando
 * multipart en `/api/predict` para curl y la página /docs.
 */
export function clasificar(baseUrl: string, base64: string): Promise<Prediccion> {
  return pedir<Prediccion>(baseUrl, '/api/predict-base64?tta=true', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ imagen_base64: base64 }),
  });
}
