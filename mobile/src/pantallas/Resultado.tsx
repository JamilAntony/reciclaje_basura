import { useCallback } from 'react';
import { ScrollView, StyleSheet, Text, View } from 'react-native';
import { Image } from 'expo-image';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { Boton } from '../componentes/Boton';
import type { Props } from '../rutas';
import { COLOR_POR_KEY, porcentaje, theme } from '../theme';

/**
 * El color del contenedor domina la pantalla porque es la única información
 * que el usuario necesita para actuar; el material y las alternativas van
 * debajo como respaldo.
 */
export function Resultado({ route, navigation }: Props<'Resultado'>) {
  const { prediccion, fotoUri } = route.params;
  const { contenedor, top, incierto } = prediccion;
  const insets = useSafeAreaInsets();

  const otraFoto = useCallback(() => navigation.replace('Camara'), [navigation]);
  const alInicio = useCallback(() => navigation.popTo('Inicio'), [navigation]);

  const ejemplos = COLOR_POR_KEY[contenedor.color]?.ejemplos;

  return (
    <View style={styles.raiz}>
      <ScrollView
        contentContainerStyle={styles.scroll}
        contentInsetAdjustmentBehavior="automatic"
      >
        <View style={[styles.tacho, { backgroundColor: contenedor.hex }]}>
          <Text style={[styles.tachoEtiqueta, { color: contenedor.texto_hex }]}>
            DEPOSITAR EN EL TACHO
          </Text>
          <Text style={[styles.tachoColor, { color: contenedor.texto_hex }]}>
            {contenedor.nombre.toUpperCase()}
          </Text>
          <Text style={[styles.tachoResiduos, { color: contenedor.texto_hex }]}>
            {contenedor.residuos}
          </Text>
        </View>

        {incierto ? (
          <View style={styles.aviso}>
            <Text style={styles.avisoTexto}>
              {prediccion.mensaje ??
                'Confianza baja: acerca más el objeto y repite la foto.'}
            </Text>
          </View>
        ) : null}

        <View style={styles.fila}>
          <Image
            source={{ uri: fotoUri }}
            style={styles.miniatura}
            contentFit="cover"
            transition={150}
          />
          <View style={styles.filaTexto}>
            <Text style={styles.material}>{prediccion.material_label}</Text>
            <Text style={styles.confianza}>
              Confianza {porcentaje.format(prediccion.confianza)}
            </Text>
            <Text style={styles.ms}>Analizado en {Math.round(prediccion.ms)} ms</Text>
          </View>
        </View>

        {prediccion.consejo ? (
          <View style={styles.consejo}>
            <Text style={styles.consejoTexto}>{prediccion.consejo}</Text>
          </View>
        ) : null}

        {ejemplos ? (
          <View style={styles.seccion}>
            <Text style={styles.seccionTitulo}>También va en este tacho</Text>
            <Text style={styles.ejemplos}>{ejemplos}</Text>
          </View>
        ) : null}

        {top.length > 1 ? (
          <View style={styles.seccion}>
            <Text style={styles.seccionTitulo}>Otras posibilidades</Text>
            <View style={styles.lista}>
              {top.slice(1).map((c) => (
                <View key={c.material} style={styles.candidato}>
                  <View style={styles.candidatoCabecera}>
                    <Text style={styles.candidatoLabel}>{c.label}</Text>
                    <Text style={styles.candidatoProb}>
                      {porcentaje.format(c.probabilidad)}
                    </Text>
                  </View>
                  <View style={styles.barraFondo}>
                    <View
                      style={[
                        styles.barra,
                        {
                          width: `${Math.max(2, c.probabilidad * 100)}%`,
                          backgroundColor: COLOR_POR_KEY[c.color]?.hex ?? theme.acento,
                        },
                      ]}
                    />
                  </View>
                </View>
              ))}
            </View>
          </View>
        ) : null}
      </ScrollView>

      <View style={[styles.acciones, { paddingBottom: insets.bottom + 16 }]}>
        <Boton
          titulo="Inicio"
          variante="secundario"
          onPress={alInicio}
          style={styles.botonAncho}
        />
        <Boton titulo="Otra foto" onPress={otraFoto} style={styles.botonAncho} />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  raiz: { flex: 1, backgroundColor: theme.fondo },
  scroll: { padding: theme.espacio, gap: 16, paddingBottom: 24 },
  tacho: {
    borderRadius: theme.radio,
    borderCurve: 'continuous',
    padding: 24,
    gap: 4,
    alignItems: 'center',
  },
  tachoEtiqueta: { fontSize: 11, letterSpacing: 1.5, opacity: 0.8 },
  tachoColor: { fontSize: 40, fontWeight: '800', letterSpacing: 1 },
  tachoResiduos: { fontSize: 15, opacity: 0.9, textAlign: 'center' },
  aviso: {
    backgroundColor: '#3A2C10',
    borderColor: theme.aviso,
    borderWidth: 1,
    borderRadius: theme.radioChico,
    borderCurve: 'continuous',
    padding: 12,
  },
  avisoTexto: { color: '#F0D79A', fontSize: 13, lineHeight: 18 },
  fila: {
    flexDirection: 'row',
    gap: 14,
    alignItems: 'center',
    backgroundColor: theme.superficie,
    borderRadius: theme.radio,
    borderCurve: 'continuous',
    padding: 14,
  },
  miniatura: {
    width: 76,
    height: 76,
    borderRadius: theme.radioChico,
    borderCurve: 'continuous',
    backgroundColor: theme.superficieAlta,
  },
  filaTexto: { flex: 1, gap: 2 },
  material: { color: theme.texto, fontSize: 22, fontWeight: '700' },
  confianza: { color: theme.textoSuave, fontSize: 14 },
  ms: { color: theme.textoSuave, fontSize: 12, opacity: 0.7 },
  consejo: {
    backgroundColor: theme.superficie,
    borderLeftColor: theme.acento,
    borderLeftWidth: 3,
    borderRadius: theme.radioChico,
    borderCurve: 'continuous',
    padding: 14,
  },
  consejoTexto: { color: theme.texto, fontSize: 14, lineHeight: 20 },
  seccion: { gap: 10 },
  seccionTitulo: {
    color: theme.textoSuave,
    fontSize: 12,
    letterSpacing: 1.2,
    textTransform: 'uppercase',
    fontWeight: '600',
  },
  ejemplos: { color: theme.textoSuave, fontSize: 14, lineHeight: 20 },
  lista: { gap: 12 },
  candidato: { gap: 6 },
  candidatoCabecera: { flexDirection: 'row', justifyContent: 'space-between' },
  candidatoLabel: { color: theme.texto, fontSize: 14 },
  candidatoProb: { color: theme.textoSuave, fontSize: 13 },
  barraFondo: {
    height: 6,
    backgroundColor: theme.superficieAlta,
    borderRadius: 3,
    borderCurve: 'continuous',
    overflow: 'hidden',
  },
  barra: { height: 6, borderRadius: 3 },
  acciones: {
    flexDirection: 'row',
    gap: 10,
    paddingHorizontal: theme.espacio,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: theme.borde,
    backgroundColor: theme.fondo,
  },
  botonAncho: { flex: 1 },
});
