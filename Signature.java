import java.lang.reflect.Array;
import java.lang.reflect.Method;
import java.lang.reflect.Field;

public class Signature {
    public static void main(String[] args) throws NoSuchMethodException, SecurityException{
        System.out.println(getSignature(Signature.class.getDeclaredMethod("getSignature",Method.class)));
    }
    public static String getSignature(Method m){
        String sig;
        try {
            Field gSig = Method.class.getDeclaredField("signature");
            gSig.setAccessible(true);
            sig = (String) gSig.get(m);
            if(sig!=null) return sig;
        } catch (IllegalAccessException | NoSuchFieldException e) { 
            // e.printStackTrace();
        }

        StringBuilder sb = new StringBuilder("(");
        for(Class<?> c : m.getParameterTypes()) 
            sb.append((sig=Array.newInstance(c, 0).toString())
                .substring(1, sig.indexOf('@')));
        return sb.append(')')
            .append(
                m.getReturnType()==void.class?"V":
                (sig=Array.newInstance(m.getReturnType(), 0).toString()).substring(1, sig.indexOf('@'))
            )
            .toString();
    }
}