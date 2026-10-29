import type { NativeStackScreenProps } from '@react-navigation/native-stack';

import type { Prediccion } from './api';

export type Rutas = {
  Inicio: undefined;
  Camara: undefined;
  Resultado: { prediccion: Prediccion; fotoUri: string };
  Guia: undefined;
  Ajustes: undefined;
};

export type Props<R extends keyof Rutas> = NativeStackScreenProps<Rutas, R>;
