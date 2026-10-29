import { useCallback, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { CameraView, useCameraPermissions, type FlashMode } from 'expo-camera';
import * as Haptics from 'expo-haptics';
import { ImageManipulator, SaveFormat } from 'expo-image-manipulator';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { alcanzable, ApiError, clasificar } from '../api';
import { Boton } from '../componentes/Boton';
import { entradaDesde, useHistorialCtx } from '../historial';
import type { Props } from '../rutas';
import { useServidorCtx } from '../ServidorContext';
import { theme } from '../theme';

/**
 * La foto se reduce antes de subirla. Una captura del S24 Ultra puede pasar de
 * 20 MB y el modelo sólo consume 224x224, así que mandarla entera sólo agrega
 * segundos de red sin ganar exactitud.
 */
const ANCHO_SUBIDA = 512;

type Fallo = { mensaje: string; detalle?: string };

/**
 * Traduce el fallo a algo accionable.
 *
 * Cuando falla la subida se comprueba aparte si el servidor contesta: si
 * responde, el problema está en la foto o en la petición, y si no responde es
 * la red. Son dos arreglos muy distintos y antes ambos salían con el mismo
 * mensaje genérico.
 */
async function explicar(err: unknown, servidor: string): Promise<Fallo> {
  if (!(err instanceof ApiError)) {
    return {
      mensaje: 'No se pudo procesar la foto. Intenta de nuevo.',
      detalle: err instanceof Error ? err.message : undefined,
    };
  }
  if (err.clase === 'servidor') {
    return { mensaje: err.message, detalle: err.causa };
  }
  if (await alcanzable(servidor)) {
    return {
      mensaje: 'El servidor responde, pero la foto no llegó a subirse.',
      detalle: err.causa,
    };
  }
  return {
    mensaje:
      'No hay conexión con el servidor. Comprueba que esté encendido y que ' +
      'el teléfono esté en la misma red Wi-Fi que la laptop.',
    detalle: err.causa,
  };
}

export function Camara({ navigation }: Props<'Camara'>) {
  const [permiso, pedirPermiso] = useCameraPermissions();
  const { servidor } = useServidorCtx();
  const { agregar } = useHistorialCtx();
  const insets = useSafeAreaInsets();

  const [ocupado, setOcupado] = useState(false);
  const [flash, setFlash] = useState<FlashMode>('off');
  const [fallo, setFallo] = useState<Fallo | null>(null);
  const camara = useRef<CameraView>(null);
  // Evita que dos toques seguidos al obturador disparen dos capturas.
  const capturando = useRef(false);

  const capturar = useCallback(async () => {
    if (capturando.current || !camara.current) return;
    capturando.current = true;
    setFallo(null);
    setOcupado(true);
    try {
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium).catch(() => {});
      const foto = await camara.current.takePictureAsync({ quality: 0.9 });
      // No es un fallo de la API, así que va como Error normal: explicar() lo
      // reporta como problema al procesar la foto, no de conexión.
      if (!foto?.uri) throw new Error('La cámara no devolvió una imagen.');

      const ref = await ImageManipulator.manipulate(foto.uri)
        .resize({ width: ANCHO_SUBIDA })
        .renderAsync();
      // base64 en la misma llamada: el backend la recibe así y nos ahorra
      // tener que leer el archivo aparte.
      const reducida = await ref.saveAsync({
        format: SaveFormat.JPEG,
        compress: 0.85,
        base64: true,
      });
      if (!reducida.base64) {
        throw new Error('No se pudo codificar la foto para enviarla.');
      }

      const prediccion = await clasificar(servidor, reducida.base64);
      agregar(entradaDesde(prediccion, reducida.uri));
      Haptics.notificationAsync(
        prediccion.incierto
          ? Haptics.NotificationFeedbackType.Warning
          : Haptics.NotificationFeedbackType.Success,
      ).catch(() => {});
      // replace: al volver desde el resultado se sale al inicio, no a una
      // cámara que ya cumplió su función.
      navigation.replace('Resultado', { prediccion, fotoUri: reducida.uri });
    } catch (err) {
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error).catch(() => {});
      setFallo(await explicar(err, servidor));
    } finally {
      capturando.current = false;
      setOcupado(false);
    }
  }, [servidor, agregar, navigation]);

  if (!permiso) {
    return (
      <View style={styles.centrado}>
        <ActivityIndicator color={theme.acento} size="large" />
      </View>
    );
  }

  if (!permiso.granted) {
    const bloqueado = !permiso.canAskAgain;
    return (
      <View style={styles.centrado}>
        <Text style={styles.tituloPermiso}>Permiso de cámara</Text>
        <Text style={styles.textoPermiso}>
          {bloqueado
            ? 'El permiso está bloqueado. Habilítalo desde los ajustes del teléfono, en la sección de permisos de Expo Go.'
            : 'Necesito la cámara para fotografiar el residuo y decirte en qué tacho va.'}
        </Text>
        {bloqueado ? null : (
          <Boton titulo="Permitir cámara" onPress={pedirPermiso} style={styles.anchoCompleto} />
        )}
        <Boton
          titulo="Volver"
          variante="secundario"
          onPress={navigation.goBack}
          style={styles.anchoCompleto}
        />
      </View>
    );
  }

  return (
    <View style={styles.raiz}>
      <CameraView ref={camara} style={styles.camara} facing="back" flash={flash} />

      <View style={[styles.cabecera, { top: insets.top + 8 }]} pointerEvents="box-none">
        <Pressable
          onPress={() => setFlash((f) => (f === 'off' ? 'on' : 'off'))}
          hitSlop={12}
          accessibilityRole="button"
          accessibilityLabel={flash === 'off' ? 'Encender el flash' : 'Apagar el flash'}
          style={({ pressed }) => [styles.chip, pressed && styles.pulsado]}
        >
          <Text style={styles.chipTexto}>
            {flash === 'off' ? 'Flash apagado' : 'Flash encendido'}
          </Text>
        </Pressable>
      </View>

      <View style={styles.guia} pointerEvents="none" />

      <View style={[styles.pie, { paddingBottom: insets.bottom + 28 }]}>
        {fallo ? (
          <View style={styles.tarjetaFallo}>
            <Text style={styles.falloTitulo}>{fallo.mensaje}</Text>
            {fallo.detalle ? (
              <Text style={styles.falloDetalle}>{fallo.detalle}</Text>
            ) : null}
          </View>
        ) : (
          <Text style={styles.ayuda}>
            {ocupado
              ? 'Clasificando...'
              : 'Encuadra un solo objeto, con buena luz y fondo despejado.'}
          </Text>
        )}

        <Pressable
          onPress={capturar}
          disabled={ocupado}
          accessibilityRole="button"
          accessibilityLabel="Tomar foto y clasificar"
          accessibilityState={{ busy: ocupado }}
          style={({ pressed }) => [
            styles.obturador,
            pressed && styles.obturadorPulsado,
            ocupado && styles.obturadorInactivo,
          ]}
        >
          {ocupado ? (
            <ActivityIndicator color={theme.fondo} size="large" />
          ) : (
            <View style={styles.obturadorNucleo} />
          )}
        </Pressable>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  raiz: { flex: 1, backgroundColor: '#000' },
  camara: { position: 'absolute', top: 0, left: 0, right: 0, bottom: 0 },
  centrado: {
    flex: 1,
    backgroundColor: theme.fondo,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 28,
    gap: 14,
  },
  anchoCompleto: { alignSelf: 'stretch' },
  tituloPermiso: {
    color: theme.texto,
    fontSize: 24,
    fontWeight: '700',
    textAlign: 'center',
  },
  textoPermiso: {
    color: theme.textoSuave,
    fontSize: 15,
    lineHeight: 22,
    textAlign: 'center',
  },
  cabecera: {
    position: 'absolute',
    left: theme.espacio,
    right: theme.espacio,
    flexDirection: 'row',
    justifyContent: 'flex-end',
  },
  chip: {
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 999,
    borderCurve: 'continuous',
    backgroundColor: 'rgba(0,0,0,0.5)',
  },
  chipTexto: { color: '#fff', fontSize: 13, fontWeight: '600' },
  pulsado: { opacity: 0.75 },
  guia: {
    position: 'absolute',
    top: '24%',
    left: '12%',
    right: '12%',
    aspectRatio: 1,
    borderWidth: 2,
    borderColor: 'rgba(255,255,255,0.5)',
    borderRadius: 24,
    borderCurve: 'continuous',
  },
  pie: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    paddingTop: 20,
    paddingHorizontal: theme.espacio,
    gap: 18,
    alignItems: 'center',
    experimental_backgroundImage:
      'linear-gradient(to bottom, rgba(0,0,0,0), rgba(0,0,0,0.8))',
  },
  ayuda: {
    color: '#fff',
    fontSize: 13,
    textAlign: 'center',
    paddingHorizontal: 16,
    opacity: 0.9,
  },
  tarjetaFallo: {
    backgroundColor: 'rgba(224,82,82,0.16)',
    borderColor: theme.error,
    borderWidth: 1,
    borderRadius: theme.radioChico,
    borderCurve: 'continuous',
    padding: 12,
    gap: 6,
    alignSelf: 'stretch',
  },
  falloTitulo: { color: '#FFD9D9', fontSize: 13, lineHeight: 19 },
  falloDetalle: {
    color: '#FFD9D9',
    fontSize: 11,
    opacity: 0.7,
    fontFamily: Platform.select({ ios: 'Menlo', android: 'monospace' }),
  },
  obturador: {
    width: 78,
    height: 78,
    borderRadius: 39,
    borderCurve: 'continuous',
    backgroundColor: 'rgba(255,255,255,0.25)',
    borderWidth: 4,
    borderColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
  obturadorPulsado: { transform: [{ scale: 0.94 }] },
  obturadorInactivo: { backgroundColor: '#fff' },
  obturadorNucleo: {
    width: 58,
    height: 58,
    borderRadius: 29,
    borderCurve: 'continuous',
    backgroundColor: '#fff',
  },
});
