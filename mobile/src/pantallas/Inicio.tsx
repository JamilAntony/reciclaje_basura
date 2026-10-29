import { useCallback } from 'react';
import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { Image } from 'expo-image';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { Boton } from '../componentes/Boton';
import { PildoraEstado } from '../componentes/PildoraEstado';
import { useHistorialCtx, type Entrada } from '../historial';
import type { Props } from '../rutas';
import { useServidorCtx } from '../ServidorContext';
import { COLORES_NTP, cuando, porcentaje, theme } from '../theme';

/**
 * Punto de entrada de la app.
 *
 * Antes se abría directo en la cámara, lo que dejaba al usuario encuadrando
 * sin saber si el servidor respondía ni dónde consultar el código de colores.
 * Aquí se ve el estado de la conexión antes de disparar, se entra a la cámara
 * a propósito y quedan a mano la guía y los escaneos previos.
 */
export function Inicio({ navigation }: Props<'Inicio'>) {
  const { servidor, estado, comprobar } = useServidorCtx();
  const { entradas } = useHistorialCtx();
  const insets = useSafeAreaInsets();

  const irACamara = useCallback(() => navigation.navigate('Camara'), [navigation]);
  const irAGuia = useCallback(() => navigation.navigate('Guia'), [navigation]);

  const recientes = entradas.slice(0, 5);

  return (
    <ScrollView
      style={styles.raiz}
      contentContainerStyle={[styles.scroll, { paddingBottom: insets.bottom + 28 }]}
      contentInsetAdjustmentBehavior="automatic"
    >
      <View style={styles.cabecera}>
        <Text style={styles.titulo}>¿En qué tacho va?</Text>
        <Text style={styles.subtitulo}>
          Toma una foto del residuo y te digo el color de contenedor que
          corresponde según la NTP 900.058.
        </Text>
      </View>

      <PildoraEstado estado={estado} servidor={servidor} onPress={comprobar} />

      <Boton
        titulo="Clasificar un residuo"
        onPress={irACamara}
        tamano="grande"
        deshabilitado={estado === 'caido'}
      />
      {estado === 'caido' ? (
        <Text style={styles.avisoCaido}>
          Enciende el backend en la laptop, o revisa la dirección en Ajustes.
        </Text>
      ) : null}

      <Pressable
        onPress={irAGuia}
        accessibilityRole="button"
        accessibilityLabel="Ver la guía de los siete contenedores"
        style={({ pressed }) => [styles.tarjetaGuia, pressed && styles.pulsado]}
      >
        <View style={styles.muestras}>
          {COLORES_NTP.map((c) => (
            <View key={c.key} style={[styles.muestra, { backgroundColor: c.hex }]} />
          ))}
        </View>
        <View style={styles.guiaTextos}>
          <Text style={styles.guiaTitulo}>Guía de los 7 contenedores</Text>
          <Text style={styles.guiaSub}>Qué va en cada color. Funciona sin conexión.</Text>
        </View>
      </Pressable>

      {recientes.length > 0 ? (
        <View style={styles.seccion}>
          <Text style={styles.seccionTitulo}>Escaneos recientes</Text>
          <View style={styles.lista}>
            {recientes.map((e) => (
              <FilaHistorial key={e.id} entrada={e} />
            ))}
          </View>
        </View>
      ) : null}
    </ScrollView>
  );
}

/** Una fila del historial. Separada para que su render no dependa de la
 *  pantalla entera. */
function FilaHistorial({ entrada }: { entrada: Entrada }) {
  return (
    <View style={styles.fila}>
      <Image
        source={{ uri: entrada.fotoUri }}
        style={styles.miniatura}
        contentFit="cover"
        transition={120}
      />
      <View style={styles.filaTextos}>
        <Text style={styles.filaMaterial}>{entrada.materialLabel}</Text>
        <Text style={styles.filaMeta}>
          {porcentaje.format(entrada.confianza)} · {cuando(entrada.ts)}
        </Text>
      </View>
      <View style={[styles.chipColor, { backgroundColor: entrada.hex }]}>
        <Text style={[styles.chipTexto, { color: entrada.textoHex }]}>
          {entrada.colorNombre}
        </Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  raiz: { flex: 1, backgroundColor: theme.fondo },
  scroll: { padding: theme.espacio, gap: 18 },
  cabecera: { gap: 8, paddingTop: 4 },
  titulo: { color: theme.texto, fontSize: 30, fontWeight: '800', lineHeight: 36 },
  subtitulo: { color: theme.textoSuave, fontSize: 15, lineHeight: 22 },
  avisoCaido: {
    color: theme.textoSuave,
    fontSize: 13,
    textAlign: 'center',
    marginTop: -8,
  },
  tarjetaGuia: {
    backgroundColor: theme.superficie,
    borderRadius: theme.radio,
    borderCurve: 'continuous',
    borderWidth: 1,
    borderColor: theme.borde,
    padding: 16,
    gap: 12,
  },
  pulsado: { opacity: 0.75 },
  muestras: { flexDirection: 'row', gap: 6 },
  muestra: {
    flex: 1,
    height: 28,
    borderRadius: 6,
    borderCurve: 'continuous',
    borderWidth: 1,
    borderColor: theme.borde,
  },
  guiaTextos: { gap: 3 },
  guiaTitulo: { color: theme.texto, fontSize: 16, fontWeight: '700' },
  guiaSub: { color: theme.textoSuave, fontSize: 13 },
  seccion: { gap: 12 },
  seccionTitulo: {
    color: theme.textoSuave,
    fontSize: 12,
    letterSpacing: 1.2,
    textTransform: 'uppercase',
    fontWeight: '600',
  },
  lista: { gap: 10 },
  fila: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    backgroundColor: theme.superficie,
    borderRadius: theme.radioChico,
    borderCurve: 'continuous',
    padding: 10,
  },
  miniatura: {
    width: 44,
    height: 44,
    borderRadius: 8,
    borderCurve: 'continuous',
    backgroundColor: theme.superficieAlta,
  },
  filaTextos: { flex: 1, gap: 2 },
  filaMaterial: { color: theme.texto, fontSize: 15, fontWeight: '600' },
  filaMeta: { color: theme.textoSuave, fontSize: 12 },
  chipColor: {
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 999,
    borderCurve: 'continuous',
  },
  chipTexto: { fontSize: 12, fontWeight: '700' },
});
