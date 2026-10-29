/**
 * Dirección del backend y su estado, disponibles en todas las pantallas.
 *
 * El estado se comprueba desde la pantalla de inicio, antes de abrir la
 * cámara: así el usuario se entera de que el servidor está caído mientras aún
 * puede hacer algo al respecto, y no después de encuadrar y disparar.
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from 'react';

import { obtenerSalud, type Salud } from './api';
import { useServidor } from './useServidor';

export type EstadoServidor = 'comprobando' | 'listo' | 'sin_modelo' | 'caido';

type Valor = ReturnType<typeof useServidor> & {
  estado: EstadoServidor;
  salud: Salud | null;
  comprobar: () => void;
};

const Ctx = createContext<Valor | null>(null);

export function ProveedorServidor({ children }: { children: ReactNode }) {
  const config = useServidor();
  const [estado, setEstado] = useState<EstadoServidor>('comprobando');
  const [salud, setSalud] = useState<Salud | null>(null);
  // Identifica la comprobación en curso: si la dirección cambia a mitad, la
  // respuesta vieja ya no debe pisar el estado de la nueva.
  const turno = useRef(0);

  const comprobar = useCallback(() => {
    if (!config.cargado) return;
    const mio = ++turno.current;
    setEstado('comprobando');
    obtenerSalud(config.servidor)
      .then((s) => {
        if (mio !== turno.current) return;
        setSalud(s);
        setEstado(s.estado === 'ok' && s.modelo ? 'listo' : 'sin_modelo');
      })
      .catch(() => {
        if (mio !== turno.current) return;
        setSalud(null);
        setEstado('caido');
      });
  }, [config.servidor, config.cargado]);

  useEffect(comprobar, [comprobar]);

  return (
    <Ctx.Provider value={{ ...config, estado, salud, comprobar }}>
      {children}
    </Ctx.Provider>
  );
}

export function useServidorCtx(): Valor {
  const v = useContext(Ctx);
  if (!v) throw new Error('useServidorCtx fuera de ProveedorServidor');
  return v;
}
