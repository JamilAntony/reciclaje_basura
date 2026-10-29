import { ScrollView, StyleSheet, Text, View } from 'react-native';

import { COLORES_NTP, theme } from '../theme';

/**
 * Los 7 contenedores de la norma, con ejemplos.
 *
 * Los datos vienen del propio paquete, no del backend: esta pantalla tiene que
 * servir justo cuando no hay laptop encendida y uno está delante del tacho.
 */
export function Guia() {
  return (
    <ScrollView
      style={styles.raiz}
      contentContainerStyle={styles.scroll}
      contentInsetAdjustmentBehavior="automatic"
    >
      <Text style={styles.intro}>
        Código de colores ampliado de la NTP 900.058, para instituciones,
        empresas y residuos no municipales.
      </Text>

      {COLORES_NTP.map((c) => (
        <View key={c.key} style={styles.tarjeta}>
          <View style={[styles.franja, { backgroundColor: c.hex }]}>
            <Text style={[styles.franjaTexto, { color: c.textoHex }]}>
              {c.nombre.toUpperCase()}
            </Text>
          </View>
          <View style={styles.cuerpo}>
            <Text style={styles.residuos}>{c.residuos}</Text>
            <Text style={styles.ejemplos}>{c.ejemplos}</Text>
          </View>
        </View>
      ))}

      <Text style={styles.nota}>
        El contenedor rojo necesita punto de acopio especial: pilas, baterías,
        aerosoles y medicinas no deben mezclarse con el resto.
      </Text>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  raiz: { flex: 1, backgroundColor: theme.fondo },
  scroll: { padding: theme.espacio, gap: 12, paddingBottom: 32 },
  intro: { color: theme.textoSuave, fontSize: 14, lineHeight: 21, marginBottom: 4 },
  tarjeta: {
    backgroundColor: theme.superficie,
    borderRadius: theme.radio,
    borderCurve: 'continuous',
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: theme.borde,
  },
  franja: { paddingHorizontal: 16, paddingVertical: 10 },
  franjaTexto: { fontSize: 15, fontWeight: '800', letterSpacing: 1 },
  cuerpo: { padding: 16, gap: 6 },
  residuos: { color: theme.texto, fontSize: 17, fontWeight: '700' },
  ejemplos: { color: theme.textoSuave, fontSize: 14, lineHeight: 20 },
  nota: {
    color: theme.textoSuave,
    fontSize: 13,
    lineHeight: 19,
    marginTop: 8,
    fontStyle: 'italic',
  },
});
