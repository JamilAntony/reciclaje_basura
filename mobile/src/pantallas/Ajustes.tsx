import { useEffect, useState } from 'react';
import { Alert, Pressable, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';

import { ApiError, obtenerSalud } from '../api';
import { Boton } from '../componentes/Boton';
import { useHistorialCtx } from '../historial';
import type { Props } from '../rutas';
import { useServidorCtx } from '../ServidorContext';
import { theme } from '../theme';

export function Ajustes({ navigation }: Props<'Ajustes'>) {
  const { servidor, esAutomatico, guardar, restaurarAutomatico, comprobar, salud } =
    useServidorCtx();
  const { entradas, limpiar } = useHistorialCtx();

  const [valor, setValor] = useState(servidor);
  const [probando, setProbando] = useState(false);
  const [resultado, setResultado] = useState<string | null>(null);
  const [ok, setOk] = useState(false);

  // Si la dirección detectada cambia (otra red, otro arranque de Expo), el
  // campo tiene que seguirla mientras el usuario no haya fijado una propia.
  useEffect(() => {
    if (esAutomatico) setValor(servidor);
  }, [servidor, esAutomatico]);

  async function probar() {
    setProbando(true);
    setResultado(null);
    try {
      const s = await obtenerSalud(valor);
      if (s.estado === 'ok' && s.modelo) {
        setOk(true);
        setResultado(
          `Conectado. Modelo ${s.modelo.arch} en ${s.modelo.device}, ` +
            `${s.modelo.clases.length} materiales.`,
        );
      } else {
        setOk(false);
        setResultado(
          s.detalle ?? 'El servidor responde pero no tiene el modelo exportado todavía.',
        );
      }
    } catch (err) {
      setOk(false);
      if (err instanceof ApiError) {
        // La causa técnica importa aquí: distingue "no hay ruta hasta la
        // laptop" de "el servidor contestó un error".
        setResultado(err.causa ? `${err.message}\n${err.causa}` : err.message);
      } else {
        setResultado('Error desconocido.');
      }
    } finally {
      setProbando(false);
    }
  }

  function confirmarLimpieza() {
    Alert.alert(
      'Borrar el historial',
      `Se eliminarán ${entradas.length} escaneos guardados en este teléfono.`,
      [
        { text: 'Cancelar', style: 'cancel' },
        { text: 'Borrar', style: 'destructive', onPress: () => void limpiar() },
      ],
    );
  }

  return (
    <ScrollView
      style={styles.raiz}
      contentContainerStyle={styles.scroll}
      contentInsetAdjustmentBehavior="automatic"
      keyboardShouldPersistTaps="handled"
    >
      <Text style={styles.titulo}>Servidor</Text>
      <Text style={styles.ayuda}>
        {esAutomatico
          ? 'Detectada a partir de la IP desde la que Expo sirve la app. Si editas y guardas, se queda fija esa dirección.'
          : 'Dirección fija. Tiene que ser la IP de la laptop en la red Wi-Fi, no localhost.'}
      </Text>

      <TextInput
        value={valor}
        onChangeText={setValor}
        placeholder="http://192.168.1.100:8000"
        placeholderTextColor={theme.textoSuave}
        autoCapitalize="none"
        autoCorrect={false}
        keyboardType="url"
        style={styles.input}
      />

      <View style={styles.botones}>
        <Boton
          titulo="Probar conexión"
          variante="secundario"
          cargando={probando}
          onPress={probar}
          style={styles.flex}
        />
        <Boton
          titulo="Guardar"
          onPress={() => {
            guardar(valor);
            comprobar();
            navigation.goBack();
          }}
          style={styles.flex}
        />
      </View>

      {esAutomatico ? null : (
        <Pressable
          onPress={() => {
            restaurarAutomatico();
            setResultado(null);
          }}
          accessibilityRole="button"
          style={({ pressed }) => [styles.enlace, pressed && styles.pulsado]}
        >
          <Text style={styles.enlaceTexto}>Volver a la detección automática</Text>
        </Pressable>
      )}

      {resultado ? (
        <Text style={[styles.resultado, ok ? styles.resOk : styles.resError]}>
          {resultado}
        </Text>
      ) : null}

      {salud?.modelo ? (
        <View style={styles.ficha}>
          <Text style={styles.fichaTitulo}>Modelo en uso</Text>
          <Text style={styles.fichaLinea}>Arquitectura: {salud.modelo.arch}</Text>
          <Text style={styles.fichaLinea}>Ejecuta en: {salud.modelo.device}</Text>
          <Text style={styles.fichaLinea}>
            Materiales: {salud.modelo.clases.length} · Límite de foto: {salud.max_mb} MB
          </Text>
        </View>
      ) : null}

      <Text style={styles.titulo}>Historial</Text>
      <Text style={styles.ayuda}>
        {entradas.length === 0
          ? 'Todavía no hay escaneos guardados.'
          : `${entradas.length} escaneos guardados en este teléfono.`}
      </Text>
      <Boton
        titulo="Borrar historial"
        variante="secundario"
        onPress={confirmarLimpieza}
        deshabilitado={entradas.length === 0}
      />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  raiz: { flex: 1, backgroundColor: theme.fondo },
  scroll: { padding: 20, gap: 12, paddingBottom: 40 },
  titulo: { color: theme.texto, fontSize: 20, fontWeight: '700', marginTop: 8 },
  ayuda: { color: theme.textoSuave, fontSize: 13, lineHeight: 19 },
  input: {
    backgroundColor: theme.superficie,
    borderColor: theme.borde,
    borderWidth: 1,
    borderRadius: theme.radioChico,
    borderCurve: 'continuous',
    color: theme.texto,
    paddingHorizontal: 14,
    paddingVertical: 12,
    fontSize: 15,
  },
  botones: { flexDirection: 'row', gap: 10 },
  flex: { flex: 1 },
  enlace: { paddingVertical: 4 },
  enlaceTexto: { color: theme.acento, fontSize: 13, fontWeight: '600' },
  pulsado: { opacity: 0.75 },
  resultado: { fontSize: 13, lineHeight: 19 },
  resOk: { color: theme.acento },
  resError: { color: theme.error },
  ficha: {
    backgroundColor: theme.superficie,
    borderRadius: theme.radioChico,
    borderCurve: 'continuous',
    borderWidth: 1,
    borderColor: theme.borde,
    padding: 14,
    gap: 4,
  },
  fichaTitulo: { color: theme.texto, fontSize: 14, fontWeight: '700', marginBottom: 2 },
  fichaLinea: { color: theme.textoSuave, fontSize: 13 },
});
