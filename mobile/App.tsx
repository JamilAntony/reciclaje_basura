import { useCallback } from 'react';
import { ActivityIndicator, Pressable, StyleSheet, Text, View } from 'react-native';
import { DarkTheme, NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { StatusBar } from 'expo-status-bar';
import { SafeAreaProvider } from 'react-native-safe-area-context';

import { Ajustes } from './src/pantallas/Ajustes';
import { Camara } from './src/pantallas/Camara';
import { Guia } from './src/pantallas/Guia';
import { Inicio } from './src/pantallas/Inicio';
import { Resultado } from './src/pantallas/Resultado';
import { ProveedorHistorial } from './src/historial';
import type { Rutas } from './src/rutas';
import { ProveedorServidor, useServidorCtx } from './src/ServidorContext';
import { theme } from './src/theme';

const Stack = createNativeStackNavigator<Rutas>();

// Stack nativo (UINavigationController / Fragment) en vez de uno en JS: trae
// gratis el gesto de volver, el botón físico de Android y las transiciones
// propias de la plataforma.
const navTheme = {
  ...DarkTheme,
  colors: {
    ...DarkTheme.colors,
    background: theme.fondo,
    card: theme.fondo,
    text: theme.texto,
    border: theme.borde,
    primary: theme.acento,
  },
};

function BotonAjustes({ onPress }: { onPress: () => void }) {
  return (
    <Pressable
      onPress={onPress}
      hitSlop={12}
      accessibilityRole="button"
      accessibilityLabel="Ajustes"
      style={({ pressed }) => pressed && styles.pulsado}
    >
      <Text style={styles.accionCabecera}>Ajustes</Text>
    </Pressable>
  );
}

function Navegacion() {
  const { cargado } = useServidorCtx();

  const opcionesInicio = useCallback(
    ({ navigation }: { navigation: { navigate: (r: 'Ajustes') => void } }) => ({
      title: 'Reciclaje NTP',
      headerRight: () => <BotonAjustes onPress={() => navigation.navigate('Ajustes')} />,
    }),
    [],
  );

  // Hasta que no se lee la dirección guardada no se sabe a qué servidor
  // apuntar, y arrancar antes haría que la pantalla de inicio comprobara una
  // dirección que está por cambiar.
  if (!cargado) {
    return (
      <View style={styles.cargando}>
        <ActivityIndicator color={theme.acento} size="large" />
      </View>
    );
  }

  return (
    <Stack.Navigator
      screenOptions={{
        headerStyle: { backgroundColor: theme.fondo },
        headerTintColor: theme.texto,
        headerTitleStyle: { fontWeight: '700' },
        headerShadowVisible: false,
        contentStyle: { backgroundColor: theme.fondo },
      }}
    >
      <Stack.Screen name="Inicio" component={Inicio} options={opcionesInicio} />
      <Stack.Screen
        name="Camara"
        component={Camara}
        options={{ title: 'Tomar foto', headerTransparent: true, headerTitle: '' }}
      />
      <Stack.Screen
        name="Resultado"
        component={Resultado}
        options={{ title: 'Resultado' }}
      />
      <Stack.Screen
        name="Guia"
        component={Guia}
        options={{ title: 'Código de colores' }}
      />
      <Stack.Screen
        name="Ajustes"
        component={Ajustes}
        options={{ title: 'Ajustes', presentation: 'modal' }}
      />
    </Stack.Navigator>
  );
}

export default function App() {
  return (
    <SafeAreaProvider>
      <StatusBar style="light" />
      <ProveedorServidor>
        <ProveedorHistorial>
          <NavigationContainer theme={navTheme}>
            <Navegacion />
          </NavigationContainer>
        </ProveedorHistorial>
      </ProveedorServidor>
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  cargando: {
    flex: 1,
    backgroundColor: theme.fondo,
    alignItems: 'center',
    justifyContent: 'center',
  },
  accionCabecera: { color: theme.acento, fontSize: 16, fontWeight: '600' },
  pulsado: { opacity: 0.6 },
});
