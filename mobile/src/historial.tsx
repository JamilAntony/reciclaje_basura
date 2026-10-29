/**
 * Historial de escaneos recientes.
 *
 * Guarda sólo lo que hace falta para redibujar la tarjeta de resultado, no la
 * respuesta completa: la foto vive en la caché del sistema y Android la borra
 * cuando necesita espacio, así que la miniatura puede desaparecer y la entrada
 * tiene que seguir siendo legible sin ella.
 */

import AsyncStorage from '@react-native-async-storage/async-storage';
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from 'react';

import type { Prediccion } from './api';

const CLAVE = 'reciclaje.historial';
const MAXIMO = 25;

export type Entrada = {
  id: string;
  ts: number;
  material: string;
  materialLabel: string;
  color: string;
  colorNombre: string;
  hex: string;
  textoHex: string;
  confianza: number;
  incierto: boolean;
  fotoUri: string;
};

export function entradaDesde(p: Prediccion, fotoUri: string): Entrada {
  return {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    ts: Date.now(),
    material: p.material,
    materialLabel: p.material_label,
    color: p.contenedor.color,
    colorNombre: p.contenedor.nombre,
    hex: p.contenedor.hex,
    textoHex: p.contenedor.texto_hex,
    confianza: p.confianza,
    incierto: p.incierto,
    fotoUri,
  };
}

export function useHistorial() {
  const [entradas, setEntradas] = useState<Entrada[]>([]);
  const [cargado, setCargado] = useState(false);

  useEffect(() => {
    let vivo = true;
    AsyncStorage.getItem(CLAVE)
      .then((txt) => {
        if (!vivo || !txt) return;
        const datos: unknown = JSON.parse(txt);
        if (Array.isArray(datos)) setEntradas(datos as Entrada[]);
      })
      .catch(() => {
        // Un historial corrupto o ilegible no debe impedir usar la app.
      })
      .finally(() => {
        if (vivo) setCargado(true);
      });
    return () => {
      vivo = false;
    };
  }, []);

  const guardar = useCallback(async (lista: Entrada[]) => {
    setEntradas(lista);
    try {
      await AsyncStorage.setItem(CLAVE, JSON.stringify(lista));
    } catch {
      // idem: la lista sigue válida en memoria para esta sesión
    }
  }, []);

  const agregar = useCallback(
    (e: Entrada) => {
      setEntradas((previas) => {
        const lista = [e, ...previas].slice(0, MAXIMO);
        AsyncStorage.setItem(CLAVE, JSON.stringify(lista)).catch(() => {});
        return lista;
      });
    },
    [],
  );

  const limpiar = useCallback(() => guardar([]), [guardar]);

  return { entradas, agregar, limpiar, cargado };
}

const Ctx = createContext<ReturnType<typeof useHistorial> | null>(null);

export function ProveedorHistorial({ children }: { children: ReactNode }) {
  return <Ctx.Provider value={useHistorial()}>{children}</Ctx.Provider>;
}

export function useHistorialCtx() {
  const v = useContext(Ctx);
  if (!v) throw new Error('useHistorialCtx fuera de ProveedorHistorial');
  return v;
}
