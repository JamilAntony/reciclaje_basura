/**
 * Dirección del backend.
 *
 * Por defecto se deduce sola: Expo ya le dijo al teléfono en qué IP está la
 * laptop (es de donde descargó el bundle), así que basta con reusar ese host
 * cambiando el puerto. Eso evita tener que reescribir la IP a mano cada vez
 * que la laptop se reconecta al Wi-Fi o cuando se comparte internet desde ella,
 * que es justo cuando la dirección cambia.
 *
 * Si hace falta apuntar a otra máquina, Ajustes guarda una dirección fija que
 * tiene prioridad sobre la detectada.
 */

import AsyncStorage from '@react-native-async-storage/async-storage';
import Constants from 'expo-constants';
import { useCallback, useEffect, useMemo, useState } from 'react';

const CLAVE = 'reciclaje.servidor';
const PUERTO_BACKEND = 8000;

/**
 * Se usa sólo si Expo no expone el host (por ejemplo en una build de
 * producción). Es un valor de ejemplo: en desarrollo nunca se llega a usar,
 * porque la dirección se detecta sola.
 */
const RESPALDO = `http://192.168.1.100:${PUERTO_BACKEND}`;

/** Host de la laptop según el servidor de desarrollo de Expo, p. ej. "192.168.1.100:8081". */
export function servidorDetectado(): string | null {
  const hostUri =
    Constants.expoConfig?.hostUri ??
    // En algunas versiones el dato viene en el manifiesto antiguo.
    (Constants.manifest2?.extra?.expoGo?.developer as unknown as undefined) ??
    null;
  if (typeof hostUri !== 'string' || hostUri.length === 0) return null;

  const host = hostUri.split('/')[0].split(':')[0];
  if (!host) return null;
  return `http://${host}:${PUERTO_BACKEND}`;
}

export function useServidor() {
  const automatico = useMemo(() => servidorDetectado() ?? RESPALDO, []);
  const [fijado, setFijado] = useState<string | null>(null);
  const [cargado, setCargado] = useState(false);

  useEffect(() => {
    let vivo = true;
    AsyncStorage.getItem(CLAVE)
      .then((guardado) => {
        if (vivo && guardado) setFijado(guardado);
      })
      .catch(() => {
        // Sin almacenamiento se sigue con la dirección detectada; no vale la
        // pena bloquear la app por esto.
      })
      .finally(() => {
        if (vivo) setCargado(true);
      });
    return () => {
      vivo = false;
    };
  }, []);

  const guardar = useCallback(async (valor: string) => {
    const limpio = valor.trim();
    setFijado(limpio);
    try {
      await AsyncStorage.setItem(CLAVE, limpio);
    } catch {
      // idem: la dirección sigue válida en memoria para esta sesión
    }
  }, []);

  const restaurarAutomatico = useCallback(async () => {
    setFijado(null);
    try {
      await AsyncStorage.removeItem(CLAVE);
    } catch {
      // idem
    }
  }, []);

  return {
    servidor: fijado ?? automatico,
    automatico,
    esAutomatico: fijado === null,
    guardar,
    restaurarAutomatico,
    cargado,
  };
}
