import { memo } from 'react';
import {
  ActivityIndicator,
  Pressable,
  StyleSheet,
  Text,
  type StyleProp,
  type ViewStyle,
} from 'react-native';

import { theme } from '../theme';

type Props = {
  titulo: string;
  onPress: () => void;
  variante?: 'principal' | 'secundario' | 'fantasma';
  tamano?: 'normal' | 'grande';
  cargando?: boolean;
  deshabilitado?: boolean;
  style?: StyleProp<ViewStyle>;
};

function BotonBase({
  titulo,
  onPress,
  variante = 'principal',
  tamano = 'normal',
  cargando = false,
  deshabilitado = false,
  style,
}: Props) {
  const inactivo = deshabilitado || cargando;

  return (
    <Pressable
      onPress={onPress}
      disabled={inactivo}
      accessibilityRole="button"
      accessibilityLabel={titulo}
      accessibilityState={{ disabled: inactivo, busy: cargando }}
      style={({ pressed }) => [
        styles.base,
        tamano === 'grande' && styles.grande,
        variante === 'principal' && styles.principal,
        variante === 'secundario' && styles.secundario,
        variante === 'fantasma' && styles.fantasma,
        pressed && styles.pulsado,
        inactivo && styles.inactivo,
        style,
      ]}
    >
      {cargando ? (
        <ActivityIndicator color={variante === 'principal' ? '#fff' : theme.texto} />
      ) : (
        <Text
          style={[
            styles.texto,
            tamano === 'grande' && styles.textoGrande,
            variante !== 'principal' && styles.textoOscuro,
          ]}
        >
          {titulo}
        </Text>
      )}
    </Pressable>
  );
}

export const Boton = memo(BotonBase);

const styles = StyleSheet.create({
  base: {
    paddingVertical: 15,
    paddingHorizontal: 20,
    borderRadius: theme.radio,
    borderCurve: 'continuous',
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: 52,
  },
  grande: { paddingVertical: 20, minHeight: 64 },
  principal: { backgroundColor: theme.acento },
  secundario: {
    backgroundColor: theme.superficieAlta,
    borderWidth: 1,
    borderColor: theme.borde,
  },
  fantasma: { backgroundColor: 'transparent' },
  pulsado: { opacity: 0.72 },
  inactivo: { opacity: 0.5 },
  texto: { color: '#fff', fontSize: 16, fontWeight: '700' },
  textoGrande: { fontSize: 18 },
  textoOscuro: { color: theme.texto },
});
