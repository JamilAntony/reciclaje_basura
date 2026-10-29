import { memo } from 'react';
import { ActivityIndicator, Pressable, StyleSheet, Text, View } from 'react-native';

import type { EstadoServidor } from '../ServidorContext';
import { theme } from '../theme';

const TEXTOS: Record<EstadoServidor, string> = {
  comprobando: 'Buscando el servidor...',
  listo: 'Servidor conectado',
  sin_modelo: 'Servidor sin modelo cargado',
  caido: 'Sin conexión con el servidor',
};

const COLORES: Record<EstadoServidor, string> = {
  comprobando: theme.textoSuave,
  listo: theme.acento,
  sin_modelo: theme.aviso,
  caido: theme.error,
};

type Props = {
  estado: EstadoServidor;
  servidor: string;
  onPress: () => void;
};

/** Estado del backend. Tocarla vuelve a comprobar; es el primer gesto útil
 *  cuando algo no responde. */
function PildoraEstadoBase({ estado, servidor, onPress }: Props) {
  return (
    <Pressable
      onPress={onPress}
      accessibilityRole="button"
      accessibilityLabel={`${TEXTOS[estado]}. Toca para volver a comprobar.`}
      style={({ pressed }) => [styles.raiz, pressed && styles.pulsado]}
    >
      {estado === 'comprobando' ? (
        <ActivityIndicator size="small" color={theme.textoSuave} />
      ) : (
        <View style={[styles.punto, { backgroundColor: COLORES[estado] }]} />
      )}
      <View style={styles.textos}>
        <Text style={[styles.titulo, { color: COLORES[estado] }]}>
          {TEXTOS[estado]}
        </Text>
        <Text style={styles.direccion} numberOfLines={1}>
          {servidor.replace(/^https?:\/\//, '')}
        </Text>
      </View>
      {estado === 'comprobando' ? null : (
        <Text style={styles.accion}>Reintentar</Text>
      )}
    </Pressable>
  );
}

export const PildoraEstado = memo(PildoraEstadoBase);

const styles = StyleSheet.create({
  raiz: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    paddingHorizontal: 14,
    paddingVertical: 12,
    borderRadius: theme.radioChico,
    borderCurve: 'continuous',
    backgroundColor: theme.superficie,
    borderWidth: 1,
    borderColor: theme.borde,
  },
  pulsado: { opacity: 0.75 },
  punto: { width: 10, height: 10, borderRadius: 5 },
  textos: { flex: 1, gap: 1 },
  titulo: { fontSize: 14, fontWeight: '600' },
  direccion: { color: theme.textoSuave, fontSize: 11 },
  accion: { color: theme.textoSuave, fontSize: 12, fontWeight: '600' },
});
